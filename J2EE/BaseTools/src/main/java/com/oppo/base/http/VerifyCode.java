/**
 * VerifyCode.java
 * com.oppo.base.http
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-6-30 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 */

package com.oppo.base.http;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Random;

import javax.servlet.http.HttpServletResponse;

import com.oppo.base.common.StringUtil;

/**
 * ClassName:VerifyCode
 * Function: 验证码图片生成
 * 
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2011-6-30 下午04:25:05
 */
public class VerifyCode {
	private int width;
	private int height;
	private int codeLength; // 验证码长度
	private int lineCount;	// 干扰线数量
	private int codeWidth;	//符号宽
	private String verifyCode;
	private BufferedImage verifyImage;
	
	
	public static final char[] RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
			.toCharArray();

	public VerifyCode() {
		this.setHeight(20);
		this.setLineCount(151);
		this.setCodeWidth(13);
		this.setCodeLength(5);
	}

	/**
	 * 生成验证码和对应图片，通过getVerifyCode()和getVerifyImage()获取
	 * 
	 * @param
	 * @return
	 */
	public void generate(HttpServletResponse response) throws IOException {
		verifyImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

		// 获取图形上下文
		Graphics g = verifyImage.getGraphics();

		// 生成随机类
		Random random = new Random();

		// 设定背景色
		g.setColor(getRandColor(200, 250));
		g.fillRect(0, 0, width, height);

		// 设定字体
		g.setFont(new Font("Times New Roman", Font.PLAIN, 18));

		// 画边框
		// g.setColor(new Color());
		// g.drawRect(0,0,width-1,height-1);

		// 随机产生155条干扰线，使图象中的认证码不易被其它程序探测到
		g.setColor(getRandColor(160, 200));
		for (int i = 0; i < lineCount; i++) {
			int x = random.nextInt(width);
			int y = random.nextInt(height);
			int xl = random.nextInt(12);
			int yl = random.nextInt(12);
			g.drawLine(x, y, x + xl, y + yl);
		}

		// 取随机产生的认证码(4位数字)
		verifyCode = StringUtil.generateRandomString(RANDOM_CHARS, codeLength);
		for (int i = 0; i < codeLength; i++) {
			String rand = String.valueOf(verifyCode.charAt(i));
			// 将认证码显示到图象中
			g.setColor(new Color(20 + random.nextInt(110), 20 + random
					.nextInt(110), 20 + random.nextInt(110)));// 调用函数出来的颜色相同，可能是因为种子太接近，所以只能直接生成
			g.drawString(rand, this.codeWidth * i + 6, 16);
		}

		// 图象生效
		g.dispose();

		response.setHeader("Pragma", "no-cache");   
		response.setHeader("Cache-Control", "no-cache");   
		response.setDateHeader("Expires", 0);   
		response.setContentType("image/jpeg");
	}
	
	public String getVerifyCode() {
		return this.verifyCode;
	}
	
	public BufferedImage getVerifyImage() {
		return verifyImage;
	}

	private Color getRandColor(int fc, int bc) {// 给定范围获得随机颜色
		Random random = new Random();

		if (fc > 255) {
			fc = 255;
		}

		if (bc > 255) {
			bc = 255;
		}

		int r = fc + random.nextInt(bc - fc);
		int g = fc + random.nextInt(bc - fc);
		int b = fc + random.nextInt(bc - fc);
		return new Color(r, g, b);
	}
	
	private void computeWidth() {
		this.width = this.codeLength * this.codeWidth + 10;
	}

	/**
	 * 获取width
	 * 
	 * @return the width
	 * @since Ver 1.0
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * 获取codeWith
	 * @return  the codeWith
	 * @since   Ver 1.0
	 */
	public int getCodeWidth() {
		return codeWidth;
	}

	/**
	 * 设置codeWith
	 * @param   codeWith    
	 * @since   Ver 1.0
	 */
	public void setCodeWidth(int codeWidth) {
		this.codeWidth = codeWidth;
		this.computeWidth();
	}

	/**
	 * 获取height
	 * 
	 * @return the height
	 * @since Ver 1.0
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * 设置height
	 * 
	 * @param height
	 * @since Ver 1.0
	 */
	public void setHeight(int height) {
		this.height = height;
	}

	/**
	 * 获取验证码长度
	 * 
	 * @return the codeLength
	 * @since Ver 1.0
	 */
	public int getCodeLength() {
		return codeLength;
	}

	/**
	 * 设置验证码长度
	 * 
	 * @param codeLength
	 * @since Ver 1.0
	 */
	public void setCodeLength(int codeLength) {
		this.codeLength = codeLength;
		this.computeWidth();
	}

	/**
	 * 获取干扰线数量
	 * 
	 * @return the lineCount
	 * @since Ver 1.0
	 */
	public int getLineCount() {
		return lineCount;
	}

	/**
	 * 设置干扰线数量
	 * 
	 * @param lineCount
	 * @since Ver 1.0
	 */
	public void setLineCount(int lineCount) {
		this.lineCount = lineCount;
	}
}
