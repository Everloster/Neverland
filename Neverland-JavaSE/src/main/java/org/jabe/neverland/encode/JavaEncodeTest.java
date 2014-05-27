package org.jabe.neverland.encode;

import java.io.UnsupportedEncodingException;

public class JavaEncodeTest {

	public static void main(String[] args) throws UnsupportedEncodingException {
		 
		String unicodeString = "哈哈哈";

		final byte[] gbkBytes = unicodeString.getBytes("GBK");

		final byte[] utf8Bytes = unicodeString.getBytes("UTF-8");

		for (byte b : gbkBytes) {
			System.out.print(Byte.toString(b));
		}
		System.out.println();
		for (byte b : utf8Bytes) {
			System.out.print(Byte.toString(b));
		}
		System.out.println();
		
		System.out.println(new String(utf8Bytes));
	}
}
