/**
 * OPPODes.java
 * com.oppo.base.security
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-18 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
 */

package com.oppo.base.security;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import com.oppo.base.file.FileOperate;

/**
 * ClassName:OPPODes 
 * Function: OPPO使用的des算法
 * 
 * 调用方式：
 * OPPODes des = new OPPODes();
 * des.encryptFile("G:\\1.txt", "23232334", "G:\\4.txt");
 * des.decryptFile("G:\\4.txt", "23232334", "G:\\5.txt");
 * 
 * @author 80036381
 * @version
 * @since Ver 1.1
 * @Date 2011-7-18 下午05:56:55
 */
public class OPPODes {

	/****** 基础参数 *****/

	// / <summary>
	// / 初始置换表
	// / </summary>
	private int[] IP_TABLE = { 57, 49, 41, 33, 25, 17, 9, 1, 59, 51, 43, 35,
			27, 19, 11, 3, 61, 53, 45, 37, 29, 21, 13, 5, 63, 55, 47, 39, 31,
			23, 15, 7, 56, 48, 40, 32, 24, 16, 8, 0, 58, 50, 42, 34, 26, 18,
			10, 2, 60, 52, 44, 36, 28, 20, 12, 4, 62, 54, 46, 38, 30, 22, 14, 6 };

	// / <summary>
	// / 逆初始置换表IP^-1
	// / </summary>
	private int[] IP_TABLE_1 = { 39, 7, 47, 15, 55, 23, 63, 31, 38, 6, 46, 14,
			54, 22, 62, 30, 37, 5, 45, 13, 53, 21, 61, 29, 36, 4, 44, 12, 52,
			20, 60, 28, 35, 3, 43, 11, 51, 19, 59, 27, 34, 2, 42, 10, 50, 18,
			58, 26, 33, 1, 41, 9, 49, 17, 57, 25, 32, 0, 40, 8, 48, 16, 56, 24 };

	// / <summary>
	// / 扩充置换表E
	// / </summary>
	private int[] E_Table = { 31, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8, 9, 10,
			11, 12, 11, 12, 13, 14, 15, 16, 15, 16, 17, 18, 19, 20, 19, 20, 21,
			22, 23, 24, 23, 24, 25, 26, 27, 28, 27, 28, 29, 30, 31, 0 };

	// / <summary>
	// / 置换函数P
	// / </summary>
	private int[] P_TABLE = { 15, 6, 19, 20, 28, 11, 27, 16, 0, 14, 22, 25, 4,
			17, 30, 9, 1, 7, 23, 13, 31, 26, 2, 8, 18, 12, 29, 5, 21, 10, 3, 24 };

	// / <summary>
	// / S盒
	// / </summary>
	private int[][][] S = {
			// S1
			new int[][] {
					new int[] { 14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9,
							0, 7 },
					new int[] { 0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5,
							3, 8 },
					new int[] { 4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10,
							5, 0 },
					new int[] { 15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0,
							6, 13 } },
			// S2
			new int[][] {
					new int[] { 15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0,
							5, 10 },
					new int[] { 3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9,
							11, 5 },
					new int[] { 0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3,
							2, 15 },
					new int[] { 13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5,
							14, 9 } },
			// S3
			new int[][] {
					new int[] { 10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4,
							2, 8 },
					new int[] { 13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11,
							15, 1 },
					new int[] { 13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10,
							14, 7 },
					new int[] { 1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5,
							2, 12 } },
			// S4
			new int[][] {
					new int[] { 7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12,
							4, 15 },
					new int[] { 13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10,
							14, 9 },
					new int[] { 10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2,
							8, 4 },
					new int[] { 3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7,
							2, 14 } },
			// S5
			new int[][] {
					new int[] { 2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0,
							14, 9 },
					new int[] { 14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9,
							8, 6 },
					new int[] { 4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3,
							0, 14 },
					new int[] { 11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4,
							5, 3 } },
			// S6
			new int[][] {
					new int[] { 12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7,
							5, 11 },
					new int[] { 10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11,
							3, 8 },
					new int[] { 9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13,
							11, 6 },
					new int[] { 4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0,
							8, 13 } },
			// S7
			new int[][] {
					new int[] { 4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10,
							6, 1 },
					new int[] { 13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15,
							8, 6 },
					new int[] { 1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5,
							9, 2 },
					new int[] { 6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2,
							3, 12 } },
			// S8
			new int[][] {
					new int[] { 13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0,
							12, 7 },
					new int[] { 1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14,
							9, 2 },
					new int[] { 7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3,
							5, 8 },
					new int[] { 2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5,
							6, 11 } } };

	// / <summary>
	// / 置换选择1
	// / </summary>
	private int[] PC_1 = { 56, 48, 40, 32, 24, 16, 8, 0, 57, 49, 41, 33, 25,
			17, 9, 1, 58, 50, 42, 34, 26, 18, 10, 2, 59, 51, 43, 35, 62, 54,
			46, 38, 30, 22, 14, 6, 61, 53, 45, 37, 29, 21, 13, 5, 60, 52, 44,
			36, 28, 20, 12, 4, 27, 19, 11, 3 };

	// / <summary>
	// / 置换选择2
	// / </summary>
	private int[] PC_2 = { 13, 16, 10, 23, 0, 4, 2, 27, 14, 5, 20, 9, 22, 18,
			11, 3, 25, 7, 15, 6, 26, 19, 12, 1, 40, 51, 30, 36, 46, 54, 29, 39,
			50, 44, 32, 46, 43, 48, 38, 55, 33, 52, 45, 41, 49, 35, 28, 31 };

	// / <summary>
	// / 对左移次数的规定
	// / </summary>
	private int[] MOVE_TIMES = { 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1 };

	private int S_LENGTH = 8;

	private void copy(byte[] sourceArray, byte[] destArray, int length) {
		copy(sourceArray, 0, destArray, 0, sourceArray.length);
	}

	private void copy(byte[] sourceArray, int sourceIndex, byte[] destArray,
			int destIndex, int length) {
		System.arraycopy(sourceArray, sourceIndex, destArray, destIndex, length);
//		for (int i = 0; i < length; i++) {
//			destArray[destIndex + i] = sourceArray[sourceIndex + i];
//		}
	}

	// 字节转换为二进制
	private byte[] byteToBit(byte ch) {
		byte[] bit = new byte[S_LENGTH];
		for (int i = 0; i < S_LENGTH; i++) {
			bit[i] = (byte) ((ch >> i) & 1);
		}

		return bit;
	}

	// 二进制转换为字节
	private byte bitToByte(byte[] bit) {
		byte ch = 0;
		for (int i = 0; i < S_LENGTH; i++) {
			ch |= (byte) (bit[i] << i);
		}

		return ch;
	}

	// 将长度为8的字符串转换为二进制位串
	private byte[] char8ToBit64(byte[] ch) {
		byte[] chs = new byte[64];
		for (int i = 0; i < S_LENGTH; i++) {
			byte[] c = byteToBit(ch[i]);
			copy(c, 0, chs, S_LENGTH * i, c.length);
		}

		return chs;
	}

	// 将二进制位串转为长度为8的字符串
	private byte[] bit64ToChar8(byte[] bit) {
		byte[] nc = new byte[S_LENGTH];

		for (int i = 0; i < S_LENGTH; i++) {
			byte[] c = new byte[S_LENGTH];
			copy(bit, i * S_LENGTH, c, 0, S_LENGTH);

			nc[i] = bitToByte(c);
		}

		return nc;
	}

	// 生成子密钥
	private byte[][] makeSubKeys(byte[] key) {
		byte[][] subKeys = new byte[16][];
		byte[] temp = new byte[56];
		temp = pc1Transform(key);
		for (int i = 0; i < 16; i++) // 16轮迭代，产生16个子密钥
		{
			rol(temp, MOVE_TIMES[i]);
			subKeys[i] = pc2Transform(temp);
		}

		return subKeys;
	}

	// 密钥置换1
	private byte[] pc1Transform(byte[] key) {
		byte[] ch = new byte[56];
		for (int i = 0; i < 56; i++) {
			ch[i] = key[PC_1[i]];
		}

		return ch;
	}

	// 密钥置换2
	private byte[] pc2Transform(byte[] key) {
		byte[] ch = new byte[48];
		for (int i = 0; i < 48; i++) {
			ch[i] = key[PC_2[i]];
		}

		return ch;
	}

	// 循环左移
	private void rol(byte[] data, int time) {
		byte[] temp = new byte[56];
		// 保存将要循环移动到右边的位
		copy(data, temp, time);
		copy(data, 28, temp, time, time);

		// 前28位移动
		copy(data, time, data, 0, 28 - time);
		copy(temp, 0, data, 28 - time, time);

		// 后28位移动
		copy(data, 28 + time, data, 28, 28 - time);
		copy(temp, time, data, 56 - time, time);
	}

	// IP置换
	private void ipTransform(byte[] data) {
		byte[] temp = new byte[64];
		for (int i = 0; i < 64; i++) {
			temp[i] = data[IP_TABLE[i]];
		}
		copy(temp, data, 64);
	}

	// IP逆置换
	private void ip1Transform(byte[] data) {
		byte[] temp = new byte[64];
		for (int i = 0; i < 64; i++) {
			temp[i] = data[IP_TABLE_1[i]];
		}
		copy(temp, data, 64);
	}

	// 扩展置换
	private void eTransform(byte[] data) {
		byte[] temp = new byte[48];
		for (int i = 0; i < 48; i++) {
			temp[i] = data[E_Table[i]];
		}
		copy(temp, data, 48);
	}

	// P置换
	private void pTransform(byte[] data) {
		byte[] temp = new byte[32];
		for (int i = 0; i < 32; i++) {
			temp[i] = data[P_TABLE[i]];
		}
		copy(temp, data, 32);
	}

	// 异或
	private void xor(byte[] r, byte[] l, int count) {
		for (int i = 0; i < count; i++) {
			r[i] ^= l[i];
		}
	}

	// S盒置换
	private void sBox(byte[] data) {
		int line, row, output;
		int cur1, cur2;
		for (int i = 0; i < S_LENGTH; i++) {
			cur1 = i * 6;
			cur2 = i << 2;

			// 计算在S盒中的行与列
			line = (data[cur1] << 1) + data[cur1 + 5];
			row = (data[cur1 + 1] << 3) + (data[cur1 + 2] << 2)
					+ (data[cur1 + 3] << 1) + data[cur1 + 4];
			output = S[i][line][row];

			// 化为二进制
			data[cur2] = (byte) ((output & 0x08) >> 3);
			data[cur2 + 1] = (byte) ((output & 0x04) >> 2);
			data[cur2 + 2] = (byte) ((output & 0x02) >> 1);
			data[cur2 + 3] = (byte) ((output & 0x01));
		}
	}

	// 加密单个分组
	private byte[] encryptBlock(byte[] plainBlock, byte[][] subKeys) {
		byte[] plainBits = char8ToBit64(plainBlock);
		byte[] temp = new byte[plainBits.length];
		byte[] copyRight = new byte[48];
		// 初始置换(IP置换)
		ipTransform(plainBits);

		int tempLen = temp.length;
		// 16轮迭代
		for (int i = 0; i < 16; i++) {
			copy(plainBits, 32, copyRight, 0, 32);
			eTransform(copyRight);
			xor(copyRight, subKeys[i], 48);
			sBox(copyRight);
			pTransform(copyRight);
			xor(plainBits, copyRight, 32);
			if (i != 15) {
				// 左右部分交换
				copy(plainBits, 0, temp, 32, 32);
				copy(plainBits, 32, temp, 0, 32);
				copy(temp, plainBits, tempLen);
			}
		}

		// 逆初始置换(IP^1置换)
		ip1Transform(plainBits);
		return bit64ToChar8(plainBits);
	}

	// 解密单个分组
	private byte[] decryptBlock(byte[] cipherBlock, byte[][] subKeys) {
		byte[] cipherBits = char8ToBit64(cipherBlock);
		byte[] temp = new byte[cipherBits.length];
		byte[] copyRight = new byte[48];

		ipTransform(cipherBits);

		int tempLen = temp.length;
		// 16轮迭代
		for (int i = 15; i >= 0; i--) {
			copy(cipherBits, 32, copyRight, 0, 32);
			eTransform(copyRight);
			xor(copyRight, subKeys[i], 48);
			sBox(copyRight);
			pTransform(copyRight);
			xor(cipherBits, copyRight, 32);
			if (i != 0) {
				// 左右部分交换
				copy(cipherBits, 0, temp, 32, 32);
				copy(cipherBits, 32, temp, 0, 32);
				copy(temp, cipherBits, tempLen);
			}
		}

		// 逆初始置换(IP^1置换)
		ip1Transform(cipherBits);
		return bit64ToChar8(cipherBits);
	}

	/**
	 * 将源文件加密后存到目标文件
	 * 
	 * @param sourceFilePath 源文件路径
	 * @param key 加密key
	 * @param destFilePath 目标文件路径
	 */
	public void encryptFile(String sourceFilePath, String key,
			String destFilePath) throws IOException {
		InputStream fis = null;
		OutputStream fos = null;
		try {
			fis = new BufferedInputStream(new FileInputStream(sourceFilePath));
			fos = new BufferedOutputStream(new FileOutputStream(destFilePath));

			encryptStream(fis, key, fos);
		} finally {
			FileOperate.close(fos);
			FileOperate.close(fis);
		}
	}

	/**
	 * 将源文件加密后存到目标文件
	 * 
	 * @param inputStream 源文件流
	 * @param key 加密key
	 * @param outputStream 目标流,建议使用带缓冲功能的OuputStream包装
	 */
	public void encryptStream(InputStream inputStream, String key,
			OutputStream outputStream) throws IOException {
		byte[] plainBlock = new byte[8];
		byte[] cipherBlock = new byte[8];
		byte[] keyBlock = new byte[8];

		// 设置密钥
		copy(key.getBytes(), keyBlock, 8);
		// 将密钥转换为二进制流
		byte[] bKey = char8ToBit64(keyBlock);
		// 生成子密钥
		byte[][] subKeys = makeSubKeys(bKey);

		int total = 0;
		int readLen; // 读取的字节数
		while ((readLen = inputStream.read(plainBlock, 0, S_LENGTH)) == S_LENGTH) {
			cipherBlock = encryptBlock(plainBlock, subKeys);
			outputStream.write(cipherBlock, 0, S_LENGTH);
			total += readLen;
		}

		if (readLen > 0) {
			total += readLen;

			// 填充
			for (int i = readLen; i < S_LENGTH - 1; i++) {
				plainBlock[i] = 0;
			}
			// 最后一个字符保存包括最后一个字符在内的锁填充的字符数量
			plainBlock[S_LENGTH - 1] = (byte) (S_LENGTH - readLen);
			cipherBlock = encryptBlock(plainBlock, subKeys);
			outputStream.write(cipherBlock, 0, S_LENGTH);
		}

		outputStream.flush();
	}

	/**
	 * 将源文件解密后存到目标文件
	 * 
	 * @param sourceFilePath 源文件路径
	 * @param key 加密key
	 * @param destFilePath 目标文件路径
	 */
	public void decryptFile(String sourceFilePath, String key,
			String destFilePath) throws IOException {
		byte[] plainBlock = new byte[8];
		byte[] cipherBlock = new byte[8];
		byte[] keyBlock = new byte[8];
		int times = 0;

		// 设置密钥
		copy(key.getBytes(), keyBlock, 8);
		// 将密钥转换为二进制流
		byte[] bKey = char8ToBit64(keyBlock);
		// 生成子密钥
		byte[][] subKeys = makeSubKeys(bKey);

		OutputStream out = null;
		InputStream in = null;
		try {
			// 获取文件长度
			File fi = new File(sourceFilePath);
			int fileLen = (int) fi.length();

			out = new BufferedOutputStream(new FileOutputStream(destFilePath));
			in = new BufferedInputStream(new FileInputStream(fi));

			while (true) {
				in.read(cipherBlock, 0, S_LENGTH);
				plainBlock = decryptBlock(cipherBlock, subKeys);
				times += S_LENGTH;
				if (times < fileLen) {
					out.write(plainBlock, 0, S_LENGTH);
				} else {
					break;
				}
			}

			// 判断末尾是否被填充
			int fillLen = plainBlock[7];
			int count = 0;
			if (fillLen < 8) {
				for (count = S_LENGTH - fillLen; count < S_LENGTH - 1; count++) {
					if (plainBlock[count] != 0) {
						break;
					}
				}
			}

			if (count == 7) {// 有填充
				out.write(plainBlock, 0, 8 - fillLen);
			} else {// 无填充
				out.write(plainBlock, 0, 8);
			}
			
			out.flush();
		} finally {
			FileOperate.close(out);
			FileOperate.close(in);
		}
	}
	
	public static void main(String[] args) throws Exception {
		OPPODes des = new OPPODes();
		long s = System.currentTimeMillis();
//		for(int i = 0; i < 3; i++) {
		des.encryptFile("G:\\OPPONewSSO2.war", "23232334", "G:\\1.txt");
		des.decryptFile("G:\\1.txt", "23232334", "G:\\OPPONewSSO2_b.war");
//		}
        long t = System.currentTimeMillis() - s;
        System.out.println(t);
	}
}
