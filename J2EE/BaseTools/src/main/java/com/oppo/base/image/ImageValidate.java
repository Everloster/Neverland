package com.oppo.base.image;

import java.awt.Point;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;

import com.oppo.base.common.OConstants;

/**
 * ClassName:ImageValidate Function: 图片验证
 * 
 * @author 80053851
 * @version
 * @since Ver 1.0
 * @Date 2011-9-5 上午10:54:03
 */
public class ImageValidate {

	/**
	 * 验证图片大小、分辨率、格式
	 * 
	 * @param src
	 *            图片文件路径
	 * @param format
	 *            验证格式数组
	 * @param min
	 *            最小分辨率
	 * @param max
	 *            最大分辨率
	 * @param deviation
	 *            分辨率允许误差
	 * @param minSize
	 *            图片大小（最小）
	 * @param maxSize
	 *            图片大小（最大）
	 * @return true:验证成功；false:验证失败
	 * @throws IOException
	 */
	public static boolean checkImage(String src, int[] format, Point min,
			Point max, int deviation, long minSize, long maxSize)
			throws IOException {
		if (src != null && "".compareTo(src) != 0) {
			src = src.trim();

			File file = new File(src);

			return checkImage(file, format, min, max, deviation, minSize,
					maxSize);
		}
		return false;
	}

	/**
	 * 验证图片大小、分辨率、格式
	 * 
	 * @param file
	 *            图片文件
	 * @param format
	 *            验证格式数组
	 * @param min
	 *            最小分辨率
	 * @param max
	 *            最大分辨率
	 * @param deviation
	 *            分辨率允许误差
	 * @param minSize
	 *            图片大小（最小）
	 * @param maxSize
	 *            图片大小（最大）
	 * @return true:验证成功；false:验证失败
	 * @throws IOException
	 */
	public static boolean checkImage(File file, int[] format, Point min,
			Point max, int deviation, long minSize, long maxSize)
			throws IOException {
		boolean b1 = checkImageTypeVailable(file, format);
		boolean b2 = checkImageResolutionVailable(file, min, max, deviation);
		boolean b3 = checkImageSizeVailable(file, minSize, maxSize);
		return b1 && b2 && b3;
	}

	/**
	 * 图片格式验证
	 * 
	 * @param file
	 *            图片文件
	 * @param format
	 *            图片限制格式0-JPG，1-BMP，2-PNG，3-GIF
	 * @return true:验证成功；false:验证失败
	 * @throws IOException
	 */
	public static boolean checkImageTypeVailable(File file, int[] format)
			throws IOException {
		if (file != null && file.exists() && format != null
				&& format.length != 0) {

			//获取格式
			String imageFormat = ImageUtil.getImageFormat(file);
			int intFormat = -1;
			if(ImageUtil.BMP.equalsIgnoreCase(imageFormat)) {
				intFormat = OConstants.IMAGE_FORMAT_BMP;
			} else if(ImageUtil.GIF.equalsIgnoreCase(imageFormat)) {
				intFormat = OConstants.IMAGE_FORMAT_GIF;
			} else if(ImageUtil.JPG.equalsIgnoreCase(imageFormat)
					|| ImageUtil.JPEG.equalsIgnoreCase(imageFormat)) {
				intFormat = OConstants.IMAGE_FORMAT_JPG;
			} else if(ImageUtil.PNG.equalsIgnoreCase(imageFormat)) {
				intFormat = OConstants.IMAGE_FORMAT_PNG;
			}
			
			for(int i = 0; i < format.length; i++) {
				if(intFormat == format[i]) {
					return true;
				}
			}
		}
		
		return false;
	}

	/**
	 * 验证图片分辨率
	 * 
	 * @param file
	 *            图片文件
	 * @param min
	 *            最小图片分辨率
	 * @param max
	 *            最大图片分辨率
	 * @param deviation
	 *            分辨率容差
	 * @return true:验证成功；false:验证失败
	 * @throws IOException
	 */
	public static boolean checkImageResolutionVailable(File file, Point min,
			Point max, int deviation) throws IOException {
		if (file != null && file.exists()) {

			FileInputStream fis = new FileInputStream(file);
			BufferedImage bufferedImg = ImageIO.read(fis);

			int imgWidth = bufferedImg.getWidth();
			int imgHeight = bufferedImg.getHeight();

			if (imgWidth != 0 && imgHeight != 0) {
				if (min == null) {
					min = new Point(0, 0);
				}
				if (max == null) {
					max = new Point(0, 0);
				}

				deviation = deviation < 0 ? 0 : deviation;

				int minW = min.x - deviation;
				int minH = min.y - deviation;
				int maxW = max.x + deviation;
				int maxH = max.y + deviation;

				minW = minW < 0 ? 0 : minW;
				minH = minH < 0 ? 0 : minH;
				maxW = maxW < 0 ? 0 : maxW;
				maxH = maxH < 0 ? 0 : maxH;

				boolean b1 = false;
				boolean b2 = false;
				// 验证宽度
				if (minW == 0 && maxW == 0) {
					b1 = true;
				} else if (minW == 0 && maxW != 0) {
					if (imgWidth <= maxW) {
						b1 = true;
					}
				} else if (minW <= maxW) {
					if (minW <= imgWidth && imgWidth <= maxW) {
						b1 = true;
					}
				}
				// 验证高度
				if (minH == 0 && maxH == 0) {
					b2 = true;
				} else if (minH == 0 && maxH != 0) {
					if (imgHeight <= maxH) {
						b2 = true;
					}
				} else if (minH <= maxH) {
					if (minH <= imgHeight && imgHeight <= maxH) {
						b2 = true;
					}
				}

				if (b1 && b2) {
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * 验证图片大小
	 * 
	 * @param file
	 *            图片文件
	 * @param size
	 *            限制大小
	 * @return
	 */
	public static boolean checkImageSizeVailable(File file, long minSize,
			long maxSize) {
		if (file != null && file.exists()) {
			long k = file.length();
			if (k != 0) {
				if (minSize == 0 && maxSize == 0) {
					return true;
				} else if (minSize == 0 && maxSize != 0) {
					if (k <= maxSize) {
						return true;
					}
				} else if (minSize == maxSize) {
					return k == minSize;
				} else if (minSize <= maxSize) {
					if (minSize <= k && k <= maxSize) {
						return true;
					}
				}
			}
		}
		return false;
	}
	
}
