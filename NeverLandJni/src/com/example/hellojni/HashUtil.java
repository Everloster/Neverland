package com.example.hellojni;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import javax.crypto.Mac;
import javax.crypto.SecretKey;
import javax.crypto.spec.SecretKeySpec;

import org.apache.commons.codec.binary.Hex;

public class HashUtil {

	private static final char[] hexDigits = "0123456789abcdef".toCharArray();

	public static String md5Hex(String str) {
		final byte[] digested = md5(str);
		if (digested == null)
			return "";
		return toHex(digested);
	}

	public static byte[] md5(String str) {
		try {
			MessageDigest digest = MessageDigest.getInstance("md5");
			byte[] digested = digest.digest(str.getBytes());
			return digested;
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
			return null;
		}
	}

	public static String md5File(File file) {
		if (!file.isFile()) {
			return null;
		}
		MessageDigest digest = null;
		FileInputStream in = null;
		byte buffer[] = new byte[1024];
		int len;
		try {
			digest = MessageDigest.getInstance("MD5");
			in = new FileInputStream(file);
			while ((len = in.read(buffer, 0, 1024)) != -1) {
				digest.update(buffer, 0, len);
			}

		} catch (Exception e) {
			return null;
		} finally {
			if (in != null) {
				try {
					in.close();
				} catch (IOException e) {
					//do nothing
				}
			}
		}
		return new String(Hex.encodeHex(digest.digest()));
	}

	public static final String toHex(byte[] bytes) {
		StringBuilder sb = new StringBuilder(2 * bytes.length);
		for (byte b : bytes) {
			sb.append(hexDigits[(b >> 4) & 0xf]).append(hexDigits[b & 0xf]);
		}
		return sb.toString();
	}

	public static String hmacSha1(String str, String keyString) {
		final byte[] keyBytes = keyString.getBytes();
		try {
			SecretKey key = new SecretKeySpec(keyBytes, "HmacSHA1");
			Mac mac = Mac.getInstance("HmacSHA1");
			mac.init(key);
			final byte[] sbs = str.getBytes();
			byte[] result = mac.doFinal(sbs);
			return toHex(result);
		} catch (NoSuchAlgorithmException e) {
			e.printStackTrace();
			return null;
		} catch (InvalidKeyException e) {
			e.printStackTrace();
			return null;
		}

	}
}
