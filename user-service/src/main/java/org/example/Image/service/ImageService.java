package org.example.Image.service;

import org.example.Image.pojo.colonoscope.Image;
import org.example.Image.pojo.colonoscope.ImageExample;

import java.util.List;

public interface ImageService {
    long countByExample(ImageExample example);

    int deleteByExample(ImageExample example);

    int deleteByPrimaryKey(Integer iid);

    int insert(Image record);

    int insertSelective(Image record);

    List<Image> selectByExample(ImageExample example);

    Image selectByPrimaryKey(Integer iid);

    int updateByExampleSelective(Image record, ImageExample example);

    int updateByExample(Image record, ImageExample example);

    int updateByPrimaryKeySelective(Image record);

    int updateByPrimaryKey(Image record);
}