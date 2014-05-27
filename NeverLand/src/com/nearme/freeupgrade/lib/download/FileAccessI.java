package com.nearme.freeupgrade.lib.download;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Serializable;

public class FileAccessI implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -7817587228118823734L;
	long nPos;
	RandomAccessFile oSavedFile;

	public FileAccessI() throws IOException {
		this("", 0);
	}

	public FileAccessI(String sName, long nPos) throws IOException {
		oSavedFile = new RandomAccessFile(sName, "rw");
		this.nPos = nPos;
		oSavedFile.seek(nPos);
	}

	public synchronized int write(byte[] b, int nStart, int nLen) throws IOException {
		int n = -1;
		oSavedFile.write(b, nStart, nLen);
		n = nLen;
		return n;
	}

}