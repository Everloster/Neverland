/**
 * ImageUtil.java
 * com.oppo.base.image
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-9-30 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.oppo.base.image;

import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

/**
 * ClassName:ImageUtil
 * Function: 图像工具类
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-9-30  下午06:22:21
 */
public class ImageUtil {
	public static final String GIF = "GIF";
	public static final String JPG = "JPG";
	public static final String JPEG = "JPEG";
	public static final String BMP = "BMP";
	public static final String PNG = "PNG";

	public static final int ROTATE_90 = 90;
	public static final int ROTATE_180 = 180;
	public static final int ROTATE_270 = 270;

	/**
	 * 根据流获取图片格式,支持InputStream,File,RandomAccessFile
	 * @param
	 * @return
	 */
	public static String getImageFormat(Object input) throws IOException {
		ImageInputStream iis = ImageIO.createImageInputStream(input);

		Iterator<ImageReader> iter = ImageIO.getImageReaders(iis);
		if (!iter.hasNext()) { // No readers found
			return null;
		}

		ImageReader reader = iter.next();
		iis.close();

		return reader.getFormatName();
	}

	/**
	 * 根据文件路径获取图片格式
	 * @param
	 * @return
	 */
	public static String getImageFormatFromFilePath(String filePath) throws IOException {
		return getImageFormat(new File(filePath));
	}

	/**
	 * 根据文件获取图片格式
	 * @param
	 * @return
	 */
	public static String getImageFormatFromFile(File file) throws IOException {
		return getImageFormat(file);
	}

	/**
	 * 压缩图片（保持原来图片的尺寸和分变率）
	 *
	 * @param oldFile 被压缩图片
	 * @param newFile 压缩后图片地址
	 * @param quality 压缩比例(0到1之间)
	 * @return
	 */
	/*
	public static boolean compressImage(String oldFile, String newFile, float quality) {

		// 验证参数的合法性
		if(StringUtil.isNullOrEmpty(oldFile) ||
				StringUtil.isNullOrEmpty(newFile)) {
			return false;
		}

		if(quality > 1) {
			return false;
		}

		try {
			// 读取原图片
			Image srcFile = ImageIO.read(new File(oldFile));
			// 取得图片的长宽
			int width = srcFile.getWidth(null);
			int height = srcFile.getHeight(null);

			BufferedImage bImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
			bImage.getGraphics().drawImage(srcFile, 0, 0, width, height, null);

			FileOutputStream out = new FileOutputStream(newFile);
			JPEGImageEncoder encoder = JPEGCodec.createJPEGEncoder(out);
		    JPEGEncodeParam jep = JPEGCodec.getDefaultJPEGEncodeParam(bImage);

		    jep.setQuality(quality, true);
		    encoder.encode(bImage, jep);

		    out.close();

		    return true;
	    } catch (FileNotFoundException e) {
	      e.printStackTrace();
	    } catch (IOException e) {
	      e.printStackTrace();
	    }

	    return false;
	}
	*/

	/**
	 * 将图片旋转指定角度，支持90,180,270
	 * @param bufferedImage 被旋转的图片
	 * @param rotateType 旋转类型
	 * @return
	 */
	public static BufferedImage rotate(BufferedImage bufferedImage, int rotateType) {
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        BufferedImage dstImage = null;
        if(rotateType == ROTATE_90 || rotateType == ROTATE_270) {
        	dstImage = new BufferedImage(height, width, bufferedImage.getType());
        } else {
        	dstImage = new BufferedImage(width, height, bufferedImage.getType());
        }

        //旋转90°
        AffineTransform affineTransform = new AffineTransform();
        if(rotateType == ROTATE_90) {
        	affineTransform.translate(height, 0.0D);
        } else if(rotateType == ROTATE_180) {
        	affineTransform.translate(width, height);
        } else if(rotateType == ROTATE_270) {
        	affineTransform.translate(0.0D, width);
        }

        affineTransform.rotate(Math.toRadians(rotateType));
        AffineTransformOp affineTransformOp = new AffineTransformOp(affineTransform, 1);

        return affineTransformOp.filter(bufferedImage, dstImage);
	}

	public static void main(String[] args) {
//		String oldFile = "D:\\image\\游戏二期-动态.png";
//		String newFile = "D:\\image\\游戏二期-动态111.png";
//		System.out.println(ImageUtil.compressImage(oldFile, newFile, 0.9f));
		File imageFile = new File("G:\\apk\\主题测试包\\27955_y2fpl.t_2.jpg");
		File destFile = new File("G:\\apk\\主题测试包\\27955_y2fpl.t_3.jpg");
		try {
			BufferedImage image = ImageIO.read(imageFile);
			if(image.getWidth() * 1.0 / image.getHeight() > 1.2) {
				BufferedImage newImage = ImageUtil.rotate(image, ImageUtil.ROTATE_90);
				ImageIO.write(newImage, "jpg", destFile);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}

