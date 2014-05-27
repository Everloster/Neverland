package com.nearme.gamecenter.jnitool;

public class JniTools {

	public static native String getFileMD5(String filePath);

	/**
	 * 本地方法 比较路径为oldPath的apk与newPath的apk之间差异，并生成patch包，存储于patchPath
	 * 
	 * @param oldPath
	 * @param newPath
	 * @param patchPath
	 * @return
	 */
	public static native int genDiff(String oldApkPath, String newApkPath,
			String patchPath);

	/**
	 * native方法 使用路径为oldApkPath的apk与路径为patchPath的补丁包，合成新的apk，并存储于newApkPath
	 * 
	 * @param oldApkPath
	 * @param newApkPath
	 * @param patchPath
	 * @return
	 */
	public static native int patch(String oldApkPath, String newApkPath,
			String patchPath);

	static {
		System.loadLibrary("GCJniTool");
	}

}
