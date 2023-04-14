package org.example.Image.service;

import java.util.List;

import org.example.Image.mapper.colonoscope.ImageMapper;
import org.example.Image.pojo.colonoscope.Image;
import org.example.Image.pojo.colonoscope.ImageExample;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class ImageServiceImpl implements ImageService {
    @Autowired
    private ImageMapper imageMapper;

    @Override
    public long countByExample(ImageExample example) {
        return imageMapper.countByExample(example);
    }

    @Override
    public int deleteByExample(ImageExample example) {
        return imageMapper.deleteByExample(example);
    }

    @Override
    public int deleteByPrimaryKey(Integer iid) {
        return imageMapper.deleteByPrimaryKey(iid);
    }

    @Override
    public int insert(Image record) {
        return imageMapper.insert(record);
    }

    @Override
    public int insertSelective(Image record) {
        return imageMapper.insertSelective(record);
    }

    @Override
    public List<Image> selectByExample(ImageExample example) {
        return imageMapper.selectByExample(example);
    }

    @Override
    public Image selectByPrimaryKey(Integer iid) {
        return imageMapper.selectByPrimaryKey(iid);
    }

    @Override
    public int updateByExampleSelective(Image record, ImageExample example) {
        return imageMapper.updateByExampleSelective(record, example);
    }

    @Override
    public int updateByExample(Image record, ImageExample example) {
        return imageMapper.updateByExample(record, example);
    }

    @Override
    public int updateByPrimaryKeySelective(Image record) {
        return imageMapper.updateByPrimaryKeySelective(record);
    }

    @Override
    public int updateByPrimaryKey(Image record) {
        return imageMapper.updateByPrimaryKey(record);
    }

}