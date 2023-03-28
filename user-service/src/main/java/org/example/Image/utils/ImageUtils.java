package org.example.Image.utils;

import org.apache.tomcat.util.codec.binary.Base64;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class ImageUtils {

    /**
     * 将BufferedImage对象转换为Base64编码的字符串
     *
     * @param image BufferedImage对象
     * @return Base64编码的字符串
     * @throws IOException 如果转换过程中出现I/O错误
     */
    public static String toBase64(BufferedImage image) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "jpeg", baos);
        byte[] bytes = baos.toByteArray();
        return Base64.encodeBase64String(bytes);
    }
}
