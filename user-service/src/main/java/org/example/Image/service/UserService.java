package org.example.Image.service;

import org.example.Image.pojo.colonoscope.User;
import org.example.Image.pojo.colonoscope.UserExample;

import java.util.List;

public interface UserService {

    long countByExample(UserExample example);

    int deleteByExample(UserExample example);

    int deleteByPrimaryKey(Integer uid);

    int insert(User record);

    int insertSelective(User record);

    List<User> selectByExample(UserExample example);

    User selectByPrimaryKey(Integer uid);

    User selectByUserName(String username);

    int updateByExampleSelective(User record, UserExample example);

    int updateByExample(User record, UserExample example);

    int updateByPrimaryKeySelective(User record);

    int updateByPrimaryKey(User record);
}


