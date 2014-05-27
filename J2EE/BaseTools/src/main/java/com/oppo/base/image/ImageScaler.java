/**
 * ImageScaler.java
 * com.oppo.base.image
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-2 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.image;

import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;

import javax.imageio.IIOImage;
import javax.imageio.ImageIO;
import javax.imageio.ImageWriteParam;
import javax.imageio.ImageWriter;
import javax.imageio.stream.FileImageOutputStream;

import com.oppo.base.common.IOUtil;
import com.oppo.base.common.StringUtil;

/**
 * ClassName:ImageScaler
 * Function: 图片缩放
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-7-2  上午10:43:03
 */
public class ImageScaler {
	private int thumbWidth = 320;
    private int thumbHeight = 480;
    private int imageType = BufferedImage.TYPE_INT_RGB;
    private int hints = Image.SCALE_SMOOTH;
    private float compressionQuality = -1;	//压缩质量,-1表示默认压缩
    private String format = "JPG";			//源图片格式
    private BufferedImage image = null;

    public ImageScaler(String imagePath) throws IOException {
    	this(imagePath, -1, -1, null);
    }
    
    public ImageScaler(String imagePath, String format) throws IOException {
    	this(imagePath, -1, -1, format);
    }
    
    /**
     * @param imagePath	原始图片路径
     * @param width 缩放后的宽,若小于1则宽度为图像原始宽
     * @param height 缩放后的长,若小于1则高度为图像原始高
     */
    public ImageScaler(String imagePath, int width, int height) throws IOException {
    	this(imagePath, width, height, null);
    }
    
    /**
     * @param imagePath	原始图片路径
     * @param width 缩放后的宽,若小于1则宽度为图像原始宽
     * @param height 缩放后的长,若小于1则高度为图像原始高
     */
    public ImageScaler(String imagePath, int width, int height, String format) throws IOException {
    	if(StringUtil.isNullOrEmpty(format)) {
	    	//尝试获取图片格式
	        String computeFormat = ImageUtil.getImageFormatFromFilePath(imagePath);
	    	if(null != computeFormat) {
	    		this.setFileFormat(computeFormat);
	    	}
    	}

    	this.image = ImageIO.read(new File(imagePath));
    	this.setImageSize(width, height);
    }
    
    public ImageScaler(File imageFile) throws IOException {
    	this(imageFile, -1, -1, null);
    }
    
    public ImageScaler(File imageFile, String format) throws IOException {
    	this(imageFile, -1, -1, format);
    }
    
    /**
     * @param imageFile	原始图片文件
     * @param width 缩放后的宽,若小于1则宽度为图像原始宽
     * @param height 缩放后的长,若小于1则高度为图像原始高
     */
    public ImageScaler(File imageFile, int width, int height) throws IOException {
    	this(imageFile, width, height, null);
    }
    
    /**
     * @param imageFile	原始图片文件
     * @param width 缩放后的宽,若小于1则宽度为图像原始宽
     * @param height 缩放后的长,若小于1则高度为图像原始高
     */
    public ImageScaler(File imageFile, int width, int height, String format) throws IOException {
    	if(StringUtil.isNullOrEmpty(format)) {
	    	//尝试获取图片格式
	        String computeFormat = ImageUtil.getImageFormatFromFile(imageFile);
	    	if(null != computeFormat) {
	    		this.setFileFormat(computeFormat);
	    	}
    	} else {
    		this.setFileFormat(format);
    	}

    	this.image = ImageIO.read(imageFile);
    	this.setImageSize(width, height);
    }
    
    public ImageScaler(InputStream inputStream) throws IOException {
    	this(inputStream, -1, -1, null);
    }
    
    public ImageScaler(InputStream inputStream, String format) throws IOException {
    	this(inputStream, -1, -1, format);
    }
    
    /**
     * @param inputStream 原始图片流
     * @param width 缩放后的宽,若小于1则宽度为图像原始宽
     * @param height 缩放后的长,若小于1则高度为图像原始高
     */
    public ImageScaler(InputStream inputStream, int width, int height) throws IOException {
    	this(inputStream, width, height, null);
    }
    
    /**
     * @param inputStream 原始图片流
     * @param width 缩放后的宽,若小于1则宽度为图像原始宽
     * @param height 缩放后的长,若小于1则高度为图像原始高
     */
    public ImageScaler(InputStream inputStream, int width, int height, String format) throws IOException {
    	byte[] data = IOUtil.getBytes(inputStream);
    	
    	if(StringUtil.isNullOrEmpty(format)) {
	    	//尝试获取图片格式
	        String computeFormat = ImageUtil.getImageFormat(new ByteArrayInputStream(data));
	    	if(null != computeFormat) {
	    		this.setFileFormat(computeFormat);
	    	}
    	}
    	
    	this.image = ImageIO.read(new ByteArrayInputStream(data));
        this.setImageSize(width, height);
    }
    
    /**
     * 设置图像大小
     * @param 
     * @return
     */
    public void setImageSize(int width, int height) {
    	this.thumbWidth = (width <= 0) ? this.image.getWidth() : width;
        this.thumbHeight = (height <= 0) ? this.image.getHeight() : height;
    }
    
    /**
     * 获取图像大小
     * @param 
     * @return
     */
    public Dimension getImageSize() {
    	return new Dimension(this.thumbWidth, this.thumbHeight);
    }
    
    public void keepAspectWithWidth() {
        this.thumbHeight = this.image.getHeight() * this.thumbWidth / this.image.getWidth();
    }
    
    public void keepAspectWithHeight() {
        this.thumbWidth = this.image.getWidth() * this.thumbHeight / this.image.getHeight();
    }
    
    /**
     * 如果宽度或者高度比指定的小则不会改变实际大小
     * @param 
     * @return
     */
    public void keepIfSmall() {
    	this.thumbWidth = Math.min(image.getWidth(), this.thumbWidth);
    	this.thumbHeight = Math.min(image.getHeight(), this.thumbHeight);
    }

    /**
     * 压缩图片并保持到指定位置
     * @param to 压缩后文件存放的位置
     */
    public void resizeWithGraphics(String to) throws IOException {
    	//图片处理
    	BufferedImage bufferImage = new BufferedImage(thumbWidth, thumbHeight, imageType);
    	Graphics2D g2d = bufferImage.createGraphics();
    	g2d.drawImage(image.getScaledInstance(thumbWidth, thumbHeight, hints), 0, 0, null);
    	g2d.dispose();
    	
    	//保存图片
    	boolean saved = false;
    	File destFile = new File(to);
    	if(compressionQuality > 0 && compressionQuality <= 1f) {
    		if(!destFile.exists() || destFile.delete()) {
		    	Iterator<ImageWriter> it = ImageIO.getImageWritersBySuffix(format);
				if (it.hasNext()) {
					FileImageOutputStream fileImageOutputStream = null;
					try {
						ImageWriter iw = (ImageWriter) it.next();
						ImageWriteParam iwp = iw.getDefaultWriteParam();
						//如果支持压缩
						if(iwp.canWriteCompressed()) {
							iwp.setCompressionMode(ImageWriteParam.MODE_EXPLICIT);
							iwp.setCompressionQuality(compressionQuality);
							
							fileImageOutputStream = new FileImageOutputStream(destFile);
							iw.setOutput(fileImageOutputStream);
							iw.write(null, new IIOImage(bufferImage, null, null), iwp);
							iw.dispose();
							
							fileImageOutputStream.flush();

							saved = true;
						}
					} finally {
						if(null != fileImageOutputStream) {
							fileImageOutputStream.close();
						}
					}
				}
    		}
    	}
    	
    	if(!saved) {
    		ImageIO.write(bufferImage, format, destFile);
    	}
    }
    
    /**
     * 获取图片对象
     * @param 
     * @return
     */
    public Image getImage() {
    	return this.image;
    }
    
    /**
     * 设置目标文件的格式
     * @param format 图片格式，如"jpg"
     */
    public void setFileFormat(String format) {
        this.format = format;
    }

    /**
	 * 获取imageType
	 * @return  the imageType
	 * @since   Ver 1.0
	 */
	public int getImageType() {
		return imageType;
	}

	/**
	 * 设置imageType
	 * @param   imageType    
	 * @since   Ver 1.0
	 */
	public void setImageType(int imageType) {
		this.imageType = imageType;
	}

	/**
	 * 获取hints
	 * @return  the hints
	 * @since   Ver 1.0
	 */
	public int getHints() {
		return hints;
	}

	/**
	 * 设置hints
	 * @param   hints    
	 * @since   Ver 1.0
	 */
	public void setHints(int hints) {
		this.hints = hints;
	}

	/**
	 * 获取压缩质量 0 - 1
	 * @return  the compressionQuality
	 * @since   Ver 1.0
	 */
	public float getCompressionQuality() {
		return compressionQuality;
	}

	/**
	 * 设置压缩质量 0 - 1
	 * @param   compressionQuality    
	 * @since   Ver 1.0
	 */
	public void setCompressionQuality(float compressionQuality) {
		this.compressionQuality = compressionQuality;
	}

	/**
	 * 设置图片是否是透明的(透明图片处理方式不同)
	 * @param   isTransparent    
	 * @since   Ver 1.0
	 */
	public void setTransparent(boolean isTransparent) {
		if(isTransparent) {
			this.setImageType(BufferedImage.TYPE_INT_ARGB);
		} else {
			this.setImageType(BufferedImage.TYPE_INT_RGB);
		}
	}

	public static void main(String[] args) {
		File f = new File("G:\\apk\\主题测试包\\27955_y2fpl.t_2.jpg");
		
		try {
			System.out.println("处理图片：" + f.getAbsolutePath());
			
			ImageScaler scaler0 = new ImageScaler(f, -1, -1, null);//png的图片需要保持不变,也可以传null,让它自己去判断
			Dimension size = scaler0.getImageSize();
			if(size.getWidth() / size.getHeight() > 1.2) {
				scaler0.setImageSize(384, 256);
			}
			//			scaler0.keepAspectWithHeight();
//			scaler0.keepAspectWithWidth();
//			scaler0.keepIfSmall();
			scaler0.setCompressionQuality(.5f);
//			scaler0.setTransparent(true);	//透明图片
			scaler0.resizeWithGraphics("G:\\apk\\主题测试包\\2.jpg");
			
		} catch (Exception e) {
			e.printStackTrace();
		}
    }
}

