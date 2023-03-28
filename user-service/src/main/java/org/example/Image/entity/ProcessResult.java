package org.example.Image.entity;

public class ProcessResult {
    private String maskImage;

    private String bboxesImage;

    public String getMaskImage() {
        return maskImage;
    }

    public void setMaskImage(String maskImage) {
        this.maskImage = maskImage;
    }

    public String getBboxesImage() {
        return bboxesImage;
    }

    public void setBboxesImage(String bboxesImage) {
        this.bboxesImage = bboxesImage;
    }

}
