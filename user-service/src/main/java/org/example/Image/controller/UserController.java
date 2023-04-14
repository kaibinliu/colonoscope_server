package org.example.Image.controller;

import org.example.Image.service.UserServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.example.Image.pojo.colonoscope.User;
import java.util.Map;
import java.util.Objects;

@RestController
@RequestMapping("/user")
@CrossOrigin(origins = "http://localhost:8080") // 允许来自http://localhost:8080的跨域请求
public class UserController {
    @Autowired
    private UserServiceImpl userService;

    @PostMapping("/register")
    public User userRegister(@RequestBody Map<String,Object> info){
        User user = new User();
        user.setUsername((String) info.get("username"));
        user.setPassword((String) info.get("password"));
        user.setTelephone((String) info.get("telephone"));
        user.setWechat((String) info.get("wechat"));
        user.setEmail((String) info.get("email"));
        user.setProvince((String) info.get("province"));
        user.setCity((String) info.get("city"));
        user.setDistrict((String) info.get("district"));
        user.setStreet((String) info.get("street"));
        int uid = userService.insert(user);
        return userService.selectByPrimaryKey(uid);
    }

    @RequestMapping("/{id}")
    public User getUserById(@PathVariable int id){
        return userService.selectByPrimaryKey(id);
    }

    @PostMapping("/login")
    public User Login(@RequestBody Map<String,Object> info){
        String username = ((String) info.get("username"));
        String password = ((String) info.get("password"));
        User user = userService.selectByUserName(username);
//        System.out.println(username+password);
//        System.out.println("密码："+user.getPassword());
        if(user == null){
//            System.out.println("获取的user为空");
            return null;
        }else if(!Objects.equals(user.getPassword(), password)){
//            System.out.println("密码错误");
            return null;
        }else{
//            System.out.println("完全正确");
            return user;
        }
    }
}
