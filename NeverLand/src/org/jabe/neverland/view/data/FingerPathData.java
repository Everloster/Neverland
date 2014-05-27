package org.jabe.neverland.view.data;

import java.io.Serializable;

import org.jabe.neverland.util.FileUtil;

/*
 * 手写或涂鸦笔记数据类
 */
public class FingerPathData implements Serializable {
	/**
	 *
	 */
	private static final long serialVersionUID = 8158037062125803807L;
	public static final int BASE_BYTES_LEN = 8;
	public int id;
	public int penColor;
	public float[] fingerPath;
	private String text;//在该手写文本后的文字

	public int getStorageSize() {
		int total = 8;
		if (fingerPath != null) {
			total += fingerPath.length * 4;
		}
		if (text != null) {
			total += text.getBytes().length;
		}
		return total;
	}

	public int toBytes(byte[] buf, int pos) {
		FileUtil.intToBytes(id, buf, pos);
		FileUtil.intToBytes(penColor, buf, pos+=4);
		int len = fingerPath.length;
		for (int j=0; j<len; j++) {
			FileUtil.intToBytes(Float.floatToIntBits(fingerPath[j]), buf, pos+=4);
		}
		return pos;
	}

	public int setWithBytes(byte[] buf, int pos, int pathLen) {
		id = FileUtil.bytesToInt(buf, pos);
		penColor = FileUtil.bytesToInt(buf, pos+=4);
		float[] datas = new float[pathLen];
		for (int j=0; j<pathLen; j++) {
			datas[j] = Float.intBitsToFloat(FileUtil.bytesToInt(buf, pos+=4));
		}
		fingerPath = datas;
		return pos;
	}
	public int bytesLength() {
		if (fingerPath != null) {
			return BASE_BYTES_LEN + fingerPath.length*4;
		}else {
			return BASE_BYTES_LEN;
		}
	}

	public int getPathLen() {
		if (fingerPath != null) {
			return fingerPath.length;
		}
		return 0;
	}

	public boolean deleteLastText() {
		if (text != null) {
			int len = text.length();
			if (len > 0) {
				text = text.substring(0, len - 1);
				return true;
			}
		}
		return false;
	}

	public void setText(String text) {
		this.text = text;
	}

	public boolean hasFingerData() {
		return this.fingerPath != null;
	}

	public String getText() {
		return text;
	}

	public boolean hasText() {
		return text!= null;
	}
}