package org.jabe.neverland.util;

import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.os.Environment;
import android.os.StatFs;
import android.text.TextUtils;

public class FileUtil {

	public static String SDCARD = null;// Environment.getExternalStorageDirectory().getPath();
	/* 当前项目产生的文件所在目录 */
	public static String ROOT_PATH = null;
	public static String TEMP_PATH = null;

	static {
		initDir();
	}

	public static void initDir() {
		SDCARD = Environment.getExternalStorageDirectory().getPath();// oppo.os.Environment.getInternalStorageDirectory().getPath();
		if (TextUtils.isEmpty(SDCARD)) {
			// SDCARD = oppo.os.Environment.getExternalSdDirectory().getPath();
		}
		ROOT_PATH = SDCARD + File.separator + "note"
				+ File.separator;
		TEMP_PATH = ROOT_PATH + "Temp/";

		if (isMounted()) {
			File file = new File(ROOT_PATH);
			if (!file.exists()) {
				file.mkdirs();
			}
			file = new File(TEMP_PATH);
			if (!file.exists()) {
				file.mkdirs();
			}
		} else {

		}
	}

	public static String getProjectFolderPath() {
		return ROOT_PATH;
	}

	public static String getTempFolder() {
		return TEMP_PATH;
	}

	/*
	 * 是否有存储卡
	 */
	public static boolean isMounted() {
		return Environment.MEDIA_MOUNTED.equals(Environment
				.getExternalStorageState());
		// boolean result = false;
		// if
		// (Environment.MEDIA_MOUNTED.equals(oppo.os.Environment.getInternalStorageState())
		// ||
		// Environment.MEDIA_MOUNTED.equals(Environment.getExternalStorageState()))
		// {
		// result = true;
		// }
		//
		// return result;
	}

	/*
	 * 存储卡路径
	 */
	public static String getStoragePath() {
		return SDCARD;
	}

	/*
	 * 在当前项目临时文件夹下创建name文件夹
	 */
	public static String createTmpFolder(String name) {
		String folder = TEMP_PATH + name + File.separator;
		File file = new File(folder);
		if (!file.exists()) {
			file.mkdirs();
		}
		return folder;
	}

	public static boolean deleteFile(String path) {
		boolean isDeleted = false;
		File file = new File(path);

		if (file.exists() && file.isFile()) {
			isDeleted = file.delete();
		}

		return isDeleted;
	}

	public static void deleteDirectory(String dirPath) {

		File dirFile = new File(dirPath);
		if (dirFile != null && dirFile.exists()) {
			deleteDirectory(dirFile);
		}

	}

	public static void deleteDirectory(File dirFile) {

		File[] files = dirFile.listFiles();
		for (File file : files) {
			if (file.isDirectory()) {
				deleteDirectory(file);
			} else if (file.isFile()) {
				file.delete();
			}
		}
		dirFile.delete();
	}

	/**
	 * 计算文件夹大小
	 *
	 * @param dir
	 * @return
	 */
	public static long getDirSize(File dir) {
		if (dir == null) {
			return 0;
		}
		if (!dir.isDirectory()) {
			return 0;
		}
		long dirSize = 0;
		File[] files = dir.listFiles();
		for (File file : files) {
			if (file.isFile()) {
				dirSize += file.length();
			} else if (file.isDirectory()) {
				dirSize += file.length();
				dirSize += getDirSize(file); // 如果遇到目录则通过递归调用继续统计
			}
		}
		return dirSize;
	}

	public static boolean isFileExist(String path) {
		if (TextUtils.isEmpty(path)) {
			return false;
		}
		File file = new File(path);
		return (file.exists());
	}

	public static byte[] readAsBytes(String path) throws OutOfMemoryError {
		FileInputStream fis = null;
		try {
			File file = new File(path);
			fis = new FileInputStream(file);
			byte[] buf = new byte[(int) file.length()];
			fis.read(buf);
			return buf;
		} catch (Exception e) {
		} finally {
			if (fis != null) {
				try {
					fis.close();
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}
		}
		return null;
	}

	public static String readAsString(String path, String coding)
			throws OutOfMemoryError {
		byte[] buf = readAsBytes(path);
		if (buf != null) {
			try {
				return new String(buf, coding);
			} catch (UnsupportedEncodingException e) {
				e.printStackTrace();
				return null;
			}
		}
		return null;
	}

	public static int bytesToInt(byte[] buf, int offset) {
		return ((buf[offset] & 0xFF) << 24) | ((buf[offset + 1] & 0xFF) << 16)
				| ((buf[offset + 2] & 0xFF) << 8) | (buf[offset + 3] & 0xFF);
	}

	public static void intToBytes(int value, byte[] buf, int offset) {
		buf[offset] = (byte) (value >> 24);
		buf[offset + 1] = (byte) (value >> 16);
		buf[offset + 2] = (byte) (value >> 8);
		buf[offset + 3] = (byte) value;
	}

	public static void save(String path, byte[] buf) {
		save(path, buf, 0, buf.length);
	}

	public static void save(String path, byte[] buf, int offset, int count) {
		FileOutputStream fos = null;
		try {
			fos = new FileOutputStream(path);
			fos.write(buf, offset, count);
		} catch (Exception e) {

		} finally {
			if (fos != null) {
				try {
					fos.close();
				} catch (IOException e1) {
					e1.printStackTrace();
				}
			}
		}
	}

	public static boolean saveBmpToFile(Bitmap bmp, String path) {

		if (bmp == null) {
			return false;
		}
		BufferedOutputStream bos = null;
		try {
			// path += ".png";
			File myCaptureFile = new File(path);
			if (myCaptureFile.exists()) {
				myCaptureFile.delete();
			}
			bos = new BufferedOutputStream(new FileOutputStream(myCaptureFile));
			bmp.compress(CompressFormat.PNG, 90, bos);
			bos.flush();
			// if (myCaptureFile.exists()) {
			// return ONE;
			// }

		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (bos != null) {
				try {
					bos.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
		return true;
	}

	public static boolean rename(String src, String dest) {
		if (src != null && dest != null) {
			File srcFile = new File(src);
			return srcFile.renameTo(new File(dest));
		} else {
			return false;
		}
	}

	public static boolean moveFile(String src, String dest) {
		return rename(src, dest);
	}

	// 单位bytes
	public static long getAvailaleSize(String path) {

		StatFs stat = new StatFs(path);
		/* 获取block的SIZE */
		long blockSize = stat.getBlockSize();
		/* 空闲的Block的数量 */
		long availableBlocks = stat.getAvailableBlocks();
		/* 返回bit大小值 */
		return (availableBlocks * blockSize); // MIB单位
		// (availableBlocks * blockSize)/1024 KIB 单位
		// (availableBlocks * blockSize)/1024 /1024 MIB单位
	}

	public static int getFileSizeOfBitmap(Bitmap bitmap) {
		if (bitmap != null) {
			return bitmap.getWidth() * bitmap.getHeight() * 4;
		}
		return 0;
	}

	public static int getFileSizeOfBitmap(int width, int height) {
		return width * height * 4;
	}

	public static long getFileSize(String path) {
		try {
			File file = new File(path);
			return file.length();
		} catch (Exception e) {
			// TODO: handle exception
		}
		return 0;
	}

	// 压缩
	  public static byte[] compress(byte[] buf) throws IOException {
	    ByteArrayOutputStream out = new ByteArrayOutputStream();
	    GZIPOutputStream gzip = new GZIPOutputStream(out);
	    gzip.write(buf);
	    gzip.close();
	    return out.toByteArray();
	  }

	  // 解压缩
	  public static byte[] uncompress(byte[] str) throws IOException {
	    ByteArrayOutputStream out = new ByteArrayOutputStream();
	    ByteArrayInputStream in = new ByteArrayInputStream(str);
	    GZIPInputStream gunzip = new GZIPInputStream(in);
	    byte[] buffer = new byte[256];
	    int n;
	    while ((n = gunzip.read(buffer)) >= 0) {
	      out.write(buffer, 0, n);
	    }
	    // toString()使用平台默认编码，也可以显式的指定如toString("UTF-8")
	    return out.toByteArray();
	  }

	  public static String parseFilename(String path) {
		  int dx = path.lastIndexOf(File.separator);
		  if (dx != -1) {
			return path.substring(dx+1);
		  }
		  return path;
	  }
}
