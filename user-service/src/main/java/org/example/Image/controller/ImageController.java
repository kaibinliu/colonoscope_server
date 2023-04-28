package org.example.Image.controller;

import org.example.Image.entity.ProcessResult;
import org.example.Image.entity.UploadResult;
import org.example.Image.pojo.colonoscope.Image;
import org.example.Image.pojo.colonoscope.ImageExample;
import org.example.Image.service.ImageServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.UUID;
import org.apache.commons.io.FilenameUtils;
import java.io.IOException;
import java.util.concurrent.TimeUnit;



@RestController
@RequestMapping("/upload")
@CrossOrigin(origins = "http://localhost:8080") // 允许来自http://localhost:8080的跨域请求
public class ImageController {

    @Value("${python.interpreter}")
    private String pythonInterpreterPath;

    @Value("${image-savepath}")
    private String image_savepath;

    @Value("${video-savepath}")
    private String video_savepath;

    @Autowired
    private ImageServiceImpl imageService;
    @PostMapping("/image")
    public ResponseEntity<UploadResult> uploadImage(@RequestParam("file") MultipartFile file) {
        try {
            // 将MultipartFile转换为BufferedImage
            BufferedImage image = ImageIO.read(file.getInputStream());

            // 生成唯一的文件名
            String fileName = generateUniqueFileName(file.getOriginalFilename());

            // 将图像保存到服务器上
            File output = new File(image_savepath + "/originalImage/" + fileName);

            if(!output.getParentFile().exists()) {
                output.getParentFile().mkdirs();
            }
            ImageIO.write(image, "jpg", output);

            // 返回结果
            UploadResult result = new UploadResult();
            result.setFileName(fileName);
            return ResponseEntity.ok(result);

//            return ResponseEntity.ok("Image processed successfully");
        } catch (IOException e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }

    @PostMapping("/video")
    public ResponseEntity<UploadResult> uploadVideo(@RequestParam("file") MultipartFile file) {
        try {
            // 获取上传的文件名
            String fileName = generateUniqueFileName(file.getOriginalFilename());
            // 设置视频文件保存路径
            String savePath = video_savepath + "/originalVideo/" + fileName;
            System.out.println("视频的保存路径为"+savePath);
            // 将视频文件保存到指定文件夹中
            file.transferTo(new File(savePath));
            // 返回上传成功的响应
            UploadResult result = new UploadResult();
            result.setFileName(fileName);
            return ResponseEntity.ok(result);
        } catch (IOException e) {
            // 返回上传失败的响应
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }


    private String generateUniqueFileName(String originalFilename) {
        String baseName = FilenameUtils.getBaseName(originalFilename);
        String extension = FilenameUtils.getExtension(originalFilename);
        String uniqueName = baseName + "-" + UUID.randomUUID().toString() + "." + extension;
        return uniqueName;
    }

    @PostMapping("/process")
    public ResponseEntity<ProcessResult> processImage(@RequestParam("fileName") String fileName, @RequestParam("uid") Integer uid) throws IOException, InterruptedException {
        try {
            // 1. 根据文件名找到对应的图像
            String image_path = image_savepath + "/originalImage/" + fileName;
            String pythonscriptpath = System.getProperty("user.dir")+"/py/process.py";
            String model_path = System.getProperty("user.dir")+"/model/unet_model-6.pth";
            System.out.println("这是图片的位置："+image_path);
            System.out.println("这是py脚本的位置："+pythonscriptpath);
            System.out.println("这是模型数据的位置："+model_path);
            ProcessResult result = processImageWithPython(pythonInterpreterPath,imageService, uid, image_path, image_savepath, model_path, pythonscriptpath);

            return new ResponseEntity<>(result, HttpStatus.OK);
        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping("/processVideo")
    public ResponseEntity<ProcessResult> processVideo(@RequestParam("fileName") String fileName, @RequestParam("uid") Integer uid) throws IOException, InterruptedException {
        try {
            // 1. 根据文件名找到对应的视频
            String video_path = video_savepath + "/originalVideo/" + fileName;
            String pythonscriptpath = System.getProperty("user.dir")+"/py/process_video.py";
            String model_path = System.getProperty("user.dir")+"/model/unet_model-6.pth";
            System.out.println("这是视频的位置："+video_path);
            System.out.println("这是py脚本的位置："+pythonscriptpath);
            System.out.println("这是模型数据的位置："+model_path);
            ProcessResult result = processVideoWithPython(pythonInterpreterPath,imageService, uid, video_path, video_savepath, model_path, pythonscriptpath);

            return new ResponseEntity<>(result, HttpStatus.OK);
        } catch (IOException e) {
            e.printStackTrace();
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @PostMapping("/history")
    public List<Image> getImageList(@RequestParam("uid") Integer uid){
        ImageExample example = new ImageExample();
        ImageExample.Criteria criteria = example.createCriteria();
        criteria.andUidEqualTo(uid);
        List<Image> list = imageService.selectByExample(example);
        System.out.println("列表中第一个图片的上传日期是："+list.get(0).getUploaddate());
        return list;
    }
    private static ProcessResult processImageWithPython(String pythonInterpreterPath,ImageServiceImpl imageService, Integer uid, String image_path, String image_savepath, String model_path, String pythonScriptPath) throws IOException, InterruptedException {

        // Create a list to hold the command and arguments
        List<String> command = new ArrayList<>();
        command.add(pythonInterpreterPath);
        command.add(pythonScriptPath);

        // Add the image path and save path to the command
        command.add(image_path);
        command.add(image_savepath);

        // Add the model path to the command
        command.add(model_path);

        // Create process builder

        System.out.println("配置的python解释器路径为："+pythonInterpreterPath);

        ProcessBuilder pb = new ProcessBuilder(command);

        // Set timeout for process execution
        long timeout = 1000; // Timeout in seconds
        TimeUnit unit = TimeUnit.SECONDS;

        // Start the process
        Process p = pb.start();

        // Wait for the process to complete
        try {
            if (!p.waitFor(timeout, unit)) {
                p.destroy(); // Destroy the process if timeout occurs
                throw new RuntimeException("Python script execution timed out after " + timeout + " " + unit.toString());
            }
            int exitCode = p.exitValue();
            if (exitCode != 0) {
                throw new RuntimeException("Python script failed with exit code: " + exitCode);
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Python script interrupted", e);
        }

        // Load the images
        File file = new File(image_path);
        String fileName = file.getName().split("\\.")[0] + ".jpeg";
        System.out.println("截取的文件名为："+fileName);
        String path1 = "/images/maskImage/" + fileName;
        String path2 = "/images/bboxesImage/" + fileName;
        System.out.println("掩模图像文件名为："+path1);
        System.out.println("标注图像文件名为："+path2);
        ProcessResult result = new ProcessResult();
        result.setMaskImage(path1);
        result.setBboxesImage(path2);
        saveProcessHistory(imageService,uid,image_path,path1,path2);
        return result;
    }

    private static ProcessResult processVideoWithPython(String pythonInterpreterPath,ImageServiceImpl imageService, Integer uid, String video_path, String video_savepath, String model_path, String pythonScriptPath) throws IOException, InterruptedException {
        // Create a list to hold the command and arguments
        List<String> command = new ArrayList<>();
        command.add(pythonInterpreterPath);
        command.add(pythonScriptPath);

        // Add the video path and save path to the command
        command.add(video_path);
        command.add(video_savepath);

        // Add the model path to the command
        command.add(model_path);

        // 创建 process builder

        System.out.println("配置的python解释器路径为："+pythonInterpreterPath);

        ProcessBuilder pb = new ProcessBuilder(command);

        // 为了程序执行设置延迟
        long timeout = 1000; // Timeout in seconds
        TimeUnit unit = TimeUnit.SECONDS;

        // 开始处理
        Process p = pb.start();

        // 等待处理完成
        try {
            if (!p.waitFor(timeout, unit)) {
                p.destroy(); // Destroy the process if timeout occurs
                throw new RuntimeException("Python script execution timed out after " + timeout + " " + unit.toString());
            }
            int exitCode = p.exitValue();
            if (exitCode != 0) {
                throw new RuntimeException("Python script failed with exit code: " + exitCode);
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Python script interrupted", e);
        }

        // 视频处理结果的路径
        File file = new File(video_path);
        String fileName = file.getName().split("\\.")[0] + ".mp4";
        String path1 = "/video/maskVideo/" + fileName;
        String path2 = "/video/bboxesVideo/" + fileName;
        ProcessResult result = new ProcessResult();
        result.setMaskImage(path1);
        result.setBboxesImage(path2);
        // 保存处理历史
        saveVideoProcessHistory(imageService,uid,video_path,path1,path2);
        return result;
    }
    public static void saveProcessHistory(ImageServiceImpl imageService, Integer uid, String imagepath, String maskpath, String boundingboxpath){
        File file = new File(imagepath);
        String imagename = file.getName();
        imagepath = "/images/originalImage/" + imagename;
        Date uploaddate = new Date();
        Image image = new Image();
        image.setUid(uid);
        image.setImagename(imagename);
        image.setImagepath(imagepath);
        image.setMaskpath(maskpath);
        image.setBoundingboxpath(boundingboxpath);
        image.setUploaddate(uploaddate);
        imageService.insert(image);
        System.out.println("uid:"+uid);
        System.out.println("imagename:"+imagename);
        System.out.println("imagepath:"+imagepath);
        System.out.println("maskpath:"+maskpath);
        System.out.println("boundingboxpath:"+boundingboxpath);
        System.out.println("uploaddate:"+uploaddate);
    }

    public static void saveVideoProcessHistory(ImageServiceImpl imageService, Integer uid, String videopath, String maskpath, String boundingboxpath){
        File file = new File(videopath);
        String videoname = file.getName();
        videopath = "/video/originalVideo/" + videoname;
        Date uploaddate = new Date();
        Image image = new Image();
        image.setUid(uid);
        image.setImagename(videoname);
        image.setImagepath(videopath);
        image.setMaskpath(maskpath);
        image.setBoundingboxpath(boundingboxpath);
        image.setUploaddate(uploaddate);
        imageService.insert(image);
        System.out.println("uid:"+uid);
        System.out.println("imagename:"+videoname);
        System.out.println("imagepath:"+videopath);
        System.out.println("maskpath:"+maskpath);
        System.out.println("boundingboxpath:"+boundingboxpath);
        System.out.println("uploaddate:"+uploaddate);
    }
}
