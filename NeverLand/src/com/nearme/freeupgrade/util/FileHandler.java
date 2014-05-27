package com.nearme.freeupgrade.util;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import android.util.Log;

/**
 * 文件处理相关
 * 
 */
public class FileHandler {

	private static FileHandler mFileHandler = new FileHandler();

	private FileHandler() {
	}

	public static FileHandler getInstance() {
		return mFileHandler;
	}

	/**
	 * 写InputStream到文件中
	 */
	public void writeFile(String path, String name, InputStream inputStream) {
		writeFile(path + name, inputStream);
	}

	/**
	 * 写InputStream到文件中
	 */
	public void writeFile(String filename, InputStream inputStream) {
		if (inputStream != null) {
			BufferedOutputStream bos = null;
			try {
				File file = new File(filename);
				bos = new BufferedOutputStream(new FileOutputStream(file));
				byte[] buffer = new byte[1024];
				int len = 0;
				while ((len = inputStream.read(buffer)) > 0) {
					bos.write(buffer, 0, len);
				}
			} catch (Exception e) {
				e.printStackTrace();
			} finally {
				if (bos != null) {
					try {
						bos.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}

				try {
					inputStream.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	/**
	 * 删除目录下的全部文件
	 */
	public void deleteFiles(String dir) {
		File dirFile = new File(dir);
		if (dirFile == null || dirFile.listFiles() == null || dirFile.listFiles().length <= 0)
			return;
		for (File file : dirFile.listFiles()) {
			deleteFile(file);
		}
	}

	/**
	 * 删除该路径文件
	 */
	public boolean deleteFile(String path) {
		File file = new File(path);
		return deleteFile(file);
	}

	/**
	 * 删除该文件
	 */
	public boolean deleteFile(File file) {
		boolean success = false;
		if (file.exists()) {
			try {
				success = file.delete();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return success;
	}

	/**
	 * 文件是否存在
	 */
	public boolean fileExist(String path) {
		File file = new File(path);
		return file.exists();
	}

	/**
	 * 删除路径（path）下，最后修改时间为时间点（time）之前的文件。
	 */
	public int deleteOverdueFiles(String path, long time) {
		return deleteFiles(new File(path), time);
	}

	/**
	 * 删除路径（dir）下，最后修改时间为时间点（time）之前的文件。
	 */
	public int deleteFiles(File dir, long time) {
		int deletedFiles = 0;
		if (dir != null && dir.isDirectory()) {
			try {
				File[] files = dir.listFiles();
				for (File child : files) {
					if (child.isDirectory()) {
						deletedFiles += deleteFiles(child, time);
					}
					if (child.lastModified() < time) {
						if (child.delete()) {
							deletedFiles++;
						}
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return deletedFiles;
	}

	public boolean makeSureDirectoryExists(String path) {
		if (path == null) {
			return false;
		}
		final File file = new File(path);
		if (!file.isDirectory() && !file.mkdirs()) {
			Log.i("makeSureDirectoryExists", "Failed to make directory:" + file.getAbsolutePath());
		}

		return file.isDirectory();
	}

	public boolean makeSureFileExist(String pathname) {
		return pathname != null && makeSureFileExist(new File(pathname));
	}

	public boolean makeSureFileExist(File file) {
		if (file == null) {
			return false;
		}
		if (!file.isFile()) {
			final File parentFile = file.getParentFile();
			if (parentFile != null && !parentFile.isDirectory()) {
				parentFile.mkdirs();
			}

			try {
				file.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		return file.isFile();
	}

}
