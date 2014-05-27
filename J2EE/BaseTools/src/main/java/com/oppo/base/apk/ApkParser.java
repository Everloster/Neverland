/**
 * ApkParser.java
 * com.oppo.base.apk
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-19 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.apk;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Map;

import org.apache.tools.zip.ZipEntry;
import org.apache.tools.zip.ZipFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import brut.androlib.AndrolibException;

import com.oppo.base.apk.entity.ApkProperty;
import com.oppo.base.apk.entity.ApkResourceProperty;
import com.oppo.base.apk.entity.ApkStringValueProperty;
import com.oppo.base.apk.entity.LockProperty;
import com.oppo.base.common.StringUtil;
import com.oppo.base.file.DirectoryOperate;
import com.oppo.base.file.FileOperate;

/**
 * ClassName:ApkParser
 * Function: apk包解析
 * 可以解析出apk包对应的jdk最低版本,包名,版本号,版本名,图标,权限列表
 *
 * File file = new File("G:\\1.apk");		//指定apk文件
 * ApkParser ap = new ApkParser("G:\\tmp");	//实例化解析类，并指定临时文件夹
 * try {
 * 		//解析指定文件，并指定图标的路径
 * 		ap.parse(file, "G:\\tmp\\" + StringUtil.generateGUID());
 * 		//获取计息到的属性
 * 		ApkProperty aProp = ap.getApkProperty();
 * 		if(null != aProp) {
 * 			String versionCode = aProp.getVersionCode();//版本号
 * 			//其他属性见ApkProperty
 * 		}
 * } catch(Exception ex){
 * 		System.out.println("parse error");
 * }
 *
 *
 * @author   80036381
 * @version
 * @see com.oppo.base.apk.entity.ApkProperty
 * @since    Ver 1.1
 * @Date	 2011-7-19  上午09:09:03
 */
public class ApkParser {
	private static final String MANIFEST_XML = "AndroidManifest.xml";
	private static final String LOCK_XML = "res/xml/lock.xml";
	private static final String CLASSES_DEX = "classes.dex";
	private static final String RESOURCES_ARSC = "resources.arsc";
	private static final String RES_DRAWABLE = "res/drawable";
	private static final String RES_DREVIEW1 = "assets/preview1";//全局主题apk预览图1
	private static final String RES_DREVIEW2 = "assets/preview2";//全局主题apk预览图1
	private static final String ID_FLAG = "@";
	//资源文件中不需保存到临时文件夹的文件
	private static final String[] FILTER_SUFFIX = {".xml", ".txt"};

	private ManifestXmlParser mXmlParser;	//manifest文件解析类
	private LockXmlParser lockXmlParser;    //res/xml/lock.xml 解析类
	private ApkClassResourceParser aResParser;	//classes.dex文件解析类
	private String tmpPath;				//临时路径，解析图标时用到
	private File tmpDir;

	private ApkProperty apkProperty;
	private LockProperty lockProperty;
	private String[] thumbPics;
	private ApkResourceProperty apkIconResourceProperty;
	private ApkResourceProperty lockThumbnailResourceProperties;
	private ApkResourceProperty lockThumbLeftResourceProperties;
	private ApkResourceProperty lockThumbRightResourceProperties;
	private Map<String, ApkResourceProperty> apkResourceMap;
	private Map<String, ApkStringValueProperty> apkStringValueMap;
	private boolean isIconResourceSaved;
	private boolean isLockThumbnailSaved;
	private boolean isLockThumbLeftSaved;
	private boolean isLockThumbRightSaved;

	private static final Logger logger = LoggerFactory.getLogger(ApkParser.class);

	public ApkParser() {
		mXmlParser = new ManifestXmlParser();
		lockXmlParser = new LockXmlParser();
		aResParser = new ApkClassResourceParser();
	}

	/**
	 * 初始化ApkParser，并指定临时路径，该临时路径用于解析apk时解析到图标，
	 * 如果不需要图标，则无需设置tmpPath
	 *
	 * @param tmpPath 临时文件夹
	 */
	public ApkParser(String tmpPath) {
		this();
		this.tmpPath = tmpPath;
	}

	/**
	 * 解析指定路径的apk文件，不保存图标
	 * @param filePath 文件路径
	 * @return
	 */
	public void parse(String filePath) throws Exception {
		parse(new File(filePath), null, null);
	}

	/**
	 * 解析指定路径的apk文件，并将图标保存为指定路径
	 * @param filePath 文件路径
	 * @param iconPath 图标的路径，若为空则不保存图标
	 * @param thumbPics  截图路径
	 * @return
	 */
	public void parse(String filePath, String iconPath, String[] thumbPics) throws Exception {
		parse(new File(filePath), iconPath, thumbPics);
	}

	/**
	 * 解析指定路径的apk文件，并将图标保存为指定路径
	 * @param filePath 文件路径
	 * @param iconPath 图标的路径，若为空则不保存图标
	 * @return
	 */
	public void parse(String filePath, String iconPath) throws Exception {
		parse(new File(filePath), iconPath, null);
	}

	/**
	 * 解析指定路径的apk文件，不保存图标
	 * @param file 文件
	 * @return
	 */
	public void parse(File file) throws Exception {
		parse(file, null, null);
	}

	/**
	 * 解析指定路径的apk文件，并将图标保存为指定路径
	 * @param file 文件
	 * @param iconPath 图标的路径，若为空则不保存图标
	 * @param thumbPics  截图路径
	 * @return
	 */
	public void parse(File file, String iconPath, String[] thumbPics) throws Exception {
		reset();

		boolean needIcon = !StringUtil.isNullOrEmpty(iconPath);
		boolean needPics = thumbPics != null && thumbPics.length > 0;
		boolean needResource =  needIcon || needPics;	//是否需要解析图片
		if(needPics){
			this.thumbPics = thumbPics;
		}

		if(needResource) {
			if(StringUtil.isNullOrEmpty(tmpPath)) {
				throw new IOException("保存图片时需要设置临时目录");
			}
		}

		ZipFile zipFile = new ZipFile(file);
		Enumeration<ZipEntry> entries = zipFile.getEntries();
		try {
			while (entries.hasMoreElements()) {
				ZipEntry ze = entries.nextElement();
				if(!ze.isDirectory()){
					String name = ze.getName();
					if(isManifest(name)) {	//解析manifest xml文件
						handleManifest(zipFile.getInputStream(ze), false, iconPath);
					} else if(needResource && isClassDex(name)) {	//解析classes.dex文件
						try {
							handleClassDex(zipFile.getInputStream(ze), name, iconPath, needIcon, needPics);
						} catch(Exception ex) {
							//处理失败时忽略
							logger.error("解析classes.dex失败:" + file.getAbsolutePath(), ex);
						}
					} else if(needResource &&(isResource(name))) {	//保存资源到临时目录
						try {
							handleResource(zipFile.getInputStream(ze), name, iconPath, needIcon, needPics);
						} catch(Exception ex) {
							//处理失败时忽略
							logger.error("解析classes.dex失败:" + file.getAbsolutePath(), ex);
						}
					} else if(isArsc(name)) {	//解析resources.arsc文件
						try {
							handleArsc(zipFile.getInputStream(ze));
						} catch(Exception ex) {
							//处理失败时忽略
							logger.error("解析classes.dex失败:" + file.getAbsolutePath(), ex);
						}
					} else if(needPics && isLockXML(name)){
						handleLockXML(zipFile.getInputStream(ze), needPics);
					}
				}
			}

			//处理完成后再修复apk属性中由id定位的信息
			if(null != apkProperty && null != apkStringValueMap) {
				//目前只处理versionName
				String versionName = apkProperty.getVersionName();
				String realVersionName = getStringValueByIdFlag(versionName);
				if(null != realVersionName) {
					apkProperty.setVersionName(realVersionName);
				}

				//
				String label = apkProperty.getLabel();
				String realLabel = getStringValueByIdFlag(label);
				if(null != realLabel) {
					apkProperty.setLabel(realLabel);
				}
			}

			if(null != lockProperty && null != apkStringValueMap){
				//处理字符串@string
				String realValue = null;
				if((realValue = getStringValueByIdFlag(lockProperty.getName())) != null){
					lockProperty.setName(realValue);
				}
				if((realValue = getStringValueByIdFlag(lockProperty.getAuthor())) != null){
					lockProperty.setAuthor(realValue);
				}
				if((realValue = getStringValueByIdFlag(lockProperty.getVersion())) != null){
					lockProperty.setVersion(realValue);
				}
				if((realValue = getStringValueByIdFlag(lockProperty.getResolution())) != null){
					lockProperty.setResolution(realValue);
				}
				if((realValue = getStringValueByIdFlag(lockProperty.getDescription())) != null){
					lockProperty.setDescription(realValue);
				}
			}

		} finally {
			if(needResource && null != tmpDir) {
				//清除临时目录
				DirectoryOperate.deleteDirectory(tmpDir, true);
			}
			zipFile.close();
		}
	}

//	/**
//	 * 解析指定路径的apk文件，并将图标保存为指定路径
//	 * @param inputStream 文件流
//	 * @param iconPath 图标的路径，若为空则不保存图标
//	 * @param thumbPics 截图保存路径，若为空则不解析截图
//	 * @return
//	 */
//	public void parse(InputStream inputStream, String iconPath, String[] thumbPics) throws Exception {
//		reset();
//
//		boolean needIcon = !StringUtil.isNullOrEmpty(iconPath);
//		boolean needPics = thumbPics != null && thumbPics.length > 0;
//		boolean needResource =  needIcon || needPics;	//是否需要解析图片
//		if(needPics){
//			this.thumbPics = thumbPics;
//		}
//
//		if(needResource) {
//			if(StringUtil.isNullOrEmpty(tmpPath)) {
//				throw new IOException("保存图片时需要设置临时目录");
//			}
//		}
//
//		ZipInputStream zipIn = null;
//		InputStream zis = null;
//		try {
//			zipIn = new ZipInputStream(new BufferedInputStream(inputStream));
//			zis = new DataInputStream(zipIn);
//			ZipEntry ze;
//			while(null != (ze = zipIn.getNextEntry())){
//				if(!ze.isDirectory()){
//					String name = ze.getName();
//					if(isManifest(name)) {	//解析manifest xml文件
//						handleManifest(zis, needIcon, iconPath);
//					} else if(needResource && isClassDex(name)) {	//解析classes.dex文件
//						try {
//							handleClassDex(zis, name, iconPath, needIcon, needPics);
//						} catch(Exception ex) {
//							//处理失败时忽略
//							ex.printStackTrace();
//						}
//					} else if(needResource && isResource(name)) {	//保存资源到临时目录
//						try {
//							handleResource(zis, name, iconPath, needIcon, needPics);
//						} catch(Exception ex) {
//							//处理失败时忽略
//							ex.printStackTrace();
//						}
//					} else if(isArsc(name)) {	//解析resources.arsc文件
//						try {
//							handleArsc(zis);
//						} catch(Exception ex) {
//							//处理失败时忽略
//							ex.printStackTrace();
//						}
//					} else if(needPics && isLockXML(name)){
//						handleLockXML(zis, needPics);
//					}
//				}
//
//				zipIn.closeEntry();
//			}
//
//			//处理完成后再修复apk属性中由id定位的信息
//			if(null != apkProperty && null != apkStringValueMap) {
//				//目前只处理versionName
//				String versionName = apkProperty.getVersionName();
//				String realVersionName = getStringValueByIdFlag(versionName);
//				if(null != realVersionName) {
//					apkProperty.setVersionName(realVersionName);
//				}
//
//				//
//				String label = apkProperty.getLabel();
//				String realLabel = getStringValueByIdFlag(label);
//				if(null != realLabel) {
//					apkProperty.setLabel(realLabel);
//				}
//			}
//
//			if(null != lockProperty && null != apkStringValueMap){
//				//处理字符串@string
//				String realValue = null;
//				if((realValue = getStringValueByIdFlag(lockProperty.getName())) != null){
//					lockProperty.setName(realValue);
//				}
//				if((realValue = getStringValueByIdFlag(lockProperty.getAuthor())) != null){
//					lockProperty.setAuthor(realValue);
//				}
//				if((realValue = getStringValueByIdFlag(lockProperty.getVersion())) != null){
//					lockProperty.setVersion(realValue);
//				}
//				if((realValue = getStringValueByIdFlag(lockProperty.getResolution())) != null){
//					lockProperty.setResolution(realValue);
//				}
//				if((realValue = getStringValueByIdFlag(lockProperty.getDescription())) != null){
//					lockProperty.setDescription(realValue);
//				}
//			}
//
//		} finally {
//			if(needResource && null != tmpDir) {	//清除临时目录
//				DirectoryOperate.deleteDirectory(tmpDir, true);
//			}
//
//			FileOperate.close(zis);
//			FileOperate.close(zipIn);
//		}
//	}

	/**
	 * 获取临时路径
	 * @return  the tmpPath
	 * @since   Ver 1.0
	 */
	public String getTmpPath() {
		return tmpPath;
	}

	/**
	 * 设置临时路径
	 * @param   tmpPath
	 * @since   Ver 1.0
	 */
	public void setTmpPath(String tmpPath) {
		this.tmpPath = tmpPath;
	}

	/**
	 * 获取apkProperty
	 * @return  the apkProperty
	 * @since   Ver 1.0
	 */
	public ApkProperty getApkProperty() {
		return apkProperty;
	}

	/**
	 * 获取lockProperty
	 * @return  the lockProperty
	 * @since   Ver 1.0
	 */
	public LockProperty getLockProperty() {
		return lockProperty;
	}

	/**
	 * 获取apkResource
	 * @return  the apkResource
	 * @since   Ver 1.0
	 */
	public ApkResourceProperty getApkIconResourceProperty() {
		return apkIconResourceProperty;
	}

	public String getDescription() {
		StringBuilder sb = new StringBuilder();
		sb.append("apk property\t[");
		sb.append((null == this.apkProperty) ? "	" : this.apkProperty.toString());
		sb.append("]\r\nicon infomation\t[");
		sb.append((null == this.apkIconResourceProperty) ? "	" : this.apkIconResourceProperty.toString());
		sb.append("]\r\n");

		return sb.toString();
	}

	protected String getStringValueByIdFlag(String idFlag) {
		if(null != idFlag && idFlag.startsWith(ID_FLAG)) {
			idFlag = "0x" + idFlag.substring(1);
			ApkStringValueProperty value = apkStringValueMap.get(idFlag);
			if(null != value) {
				return value.getValue();
			}
		}

		return null;
	}

	//处理manifest
	protected void handleManifest(InputStream inputStream, boolean needIcon, String iconPath) throws Exception {
		mXmlParser.parse(inputStream);
		apkProperty = mXmlParser.getApkProperty();

		if(needIcon && null != apkResourceMap) {
			apkIconResourceProperty = apkResourceMap.get(apkProperty.getIconAddress());
			//拷贝文件
			File[] resFiles = getTmpDir().listFiles();
			int resFilesLen = resFiles.length;
			for(int i = 0; i < resFilesLen; i++) {
				String fileName = resFiles[i].getName();
				int dotIndex = fileName.lastIndexOf('.');
				if(apkIconResourceProperty.getResource().equals(fileName.substring(0, dotIndex))){
					FileOperate.copyFile(resFiles[i].getAbsolutePath(), iconPath);
					isIconResourceSaved = true;
					break;
				}
			}
		}
	}

	//处理lock.xml
	protected void handleLockXML(InputStream inputStream, boolean needPics) throws Exception {
		lockXmlParser.parse(inputStream);
		lockProperty = lockXmlParser.getLockProperty();

		if(needPics && null != apkResourceMap) {
			lockThumbnailResourceProperties = apkResourceMap.get(lockProperty.getThumbnail());
			lockThumbLeftResourceProperties = apkResourceMap.get(lockProperty.getThumbLeft());
			lockThumbRightResourceProperties = apkResourceMap.get(lockProperty.getThumbRight());
			//拷贝文件
			File[] resFiles = getTmpDir().listFiles();
			int resFilesLen = resFiles.length;
			for(int i = 0; i < resFilesLen; i++) {
				String fileName = resFiles[i].getName();
				int dotIndex = fileName.lastIndexOf('.');
				if(lockThumbnailResourceProperties.getResource().equals(fileName.substring(0, dotIndex))){
					if(thumbPics.length > 0){
						FileOperate.copyFile(resFiles[i].getAbsolutePath(), thumbPics[0]);
						isLockThumbnailSaved = true;
					}
				}else if(lockThumbLeftResourceProperties.getResource().equals(fileName.substring(0, dotIndex))){
					if(thumbPics.length > 1){
						FileOperate.copyFile(resFiles[i].getAbsolutePath(), thumbPics[1]);
						isLockThumbLeftSaved = true;
					}
				}else if(lockThumbRightResourceProperties.getResource().equals(fileName.substring(0, dotIndex))){
					if(thumbPics.length > 2){
						FileOperate.copyFile(resFiles[i].getAbsolutePath(), thumbPics[2]);
						isLockThumbRightSaved = true;
					}
				}
			}
		}
	}

	//处理classes.dex中的资源文件
	protected void handleClassDex(InputStream inputStream, String resourceFullName, String iconPath, boolean needIcon, boolean needPics) throws Exception {

		File tmpFile = new File(tmpPath, StringUtil.generateGUID());
		FileOperate.saveStreamToFile(tmpFile, inputStream);

		if(needPics || (needIcon && null == apkProperty)){
			aResParser.parseResource(tmpFile, null);
			apkResourceMap = aResParser.getAllResource();
		}

		if(apkResourceMap != null){
			if(needIcon && apkProperty != null){
				apkIconResourceProperty = apkResourceMap.get(apkProperty.getIconAddress());
			}
			if(needPics && lockProperty != null){
				lockThumbnailResourceProperties = apkResourceMap.get(lockProperty.getThumbnail());
				lockThumbLeftResourceProperties = apkResourceMap.get(lockProperty.getThumbLeft());
				lockThumbRightResourceProperties = apkResourceMap.get(lockProperty.getThumbRight());
			}
		}else{
			if(needIcon && apkProperty != null){
				apkIconResourceProperty = aResParser.search(tmpFile, apkProperty.getPackageName(), apkProperty.getIconAddress());
			}
		}

		if(apkIconResourceProperty != null || lockThumbnailResourceProperties != null || lockThumbLeftResourceProperties != null ||
				lockThumbRightResourceProperties != null){
			//将临时文件夹中的文件拷贝到指定路径   没有考虑图片格式
			File[] resFiles = getTmpDir().listFiles();
			int resFilesLen = resFiles.length;
			for(int i = 0; i < resFilesLen; i++) {
				String fileName = resFiles[i].getName();
				int dotIndex = fileName.lastIndexOf('.');
				if(apkIconResourceProperty != null && apkIconResourceProperty.getResource().equals(fileName.substring(0, dotIndex))){
					FileOperate.copyFile(resFiles[i].getAbsolutePath(), iconPath);
					isIconResourceSaved = true;
				}else if(lockThumbnailResourceProperties != null && lockThumbnailResourceProperties.getResource().equals(fileName.substring(0, dotIndex))){
					if(thumbPics.length > 0){
						FileOperate.copyFile(resFiles[i].getAbsolutePath(), thumbPics[0]);
						isLockThumbnailSaved = true;
					}
				}else if(lockThumbLeftResourceProperties != null && lockThumbLeftResourceProperties.getResource().equals(fileName.substring(0, dotIndex))){
					if(thumbPics.length > 1){
						FileOperate.copyFile(resFiles[i].getAbsolutePath(), thumbPics[1]);
						isLockThumbLeftSaved = true;
					}
				}else if(lockThumbRightResourceProperties != null && lockThumbRightResourceProperties.getResource().equals(fileName.substring(0, dotIndex))){
					if(thumbPics.length > 2){
						FileOperate.copyFile(resFiles[i].getAbsolutePath(), thumbPics[2]);
						isLockThumbRightSaved = true;
					}
				}
			}
		}
	}

	//处理资源文件
	protected void handleResource(InputStream inputStream, String resourceFullName, String iconPath, boolean needIcon, boolean needPics) throws IOException {

		if((needIcon && !isIconResourceSaved) || (needPics && (!isLockThumbnailSaved || !isLockThumbLeftSaved || !isLockThumbRightSaved))){
			int splitIndex = resourceFullName.lastIndexOf('/');
			int dotIndex = resourceFullName.lastIndexOf('.');
			if(splitIndex < 0 || dotIndex <= splitIndex) {
				return;
			}
			String resourceName = resourceFullName.substring(splitIndex + 1, dotIndex);
			String suffix = resourceFullName.substring(dotIndex);

			boolean isResourceFile = true;
			for(int i = 0; i < FILTER_SUFFIX.length; i++) {
				if(FILTER_SUFFIX[i].equalsIgnoreCase(suffix)) {
					isResourceFile = false;
					break;
				}
			}
			if(isResourceFile){
				if(null != apkIconResourceProperty && !isIconResourceSaved && resourceName.equals(apkIconResourceProperty.getResource())) {
					//保存到iconPath中
					FileOperate.saveStreamToFile(iconPath, inputStream);
					isIconResourceSaved = true;
				}else if(lockThumbnailResourceProperties != null && !isLockThumbnailSaved && resourceName.equals(lockThumbnailResourceProperties.getResource())){
					if(thumbPics.length > 0){
						FileOperate.saveStreamToFile(thumbPics[0], inputStream);
						isLockThumbnailSaved = true;
					}
				}else if(lockThumbLeftResourceProperties != null && !isLockThumbLeftSaved && resourceName.equals(lockThumbLeftResourceProperties.getResource())){
					if(thumbPics.length > 1){
						FileOperate.saveStreamToFile(thumbPics[1], inputStream);
						isLockThumbLeftSaved = true;
					}
				}else if(lockThumbRightResourceProperties != null && !isLockThumbRightSaved && resourceName.equals(lockThumbRightResourceProperties.getResource())){
					if(thumbPics.length > 2){
						FileOperate.saveStreamToFile(thumbPics[2], inputStream);
						isLockThumbRightSaved = true;
					}
				} else {
					//保存资源文件到临时文件夹里
					File file = new File(getTmpDir(), resourceName + suffix);
					FileOperate.saveStreamToFile(file, inputStream);
				}
			}
		}
	}

	protected void handleArsc(InputStream in) throws AndrolibException {
		apkStringValueMap = ApkArscResourceParser.getProperties(in);
	}

	private File getTmpDir() throws IOException {
		if(null == tmpDir) {
			//创建临时目录存放资源
			tmpDir = new File(tmpPath, StringUtil.generateGUID());
			if(!tmpDir.mkdirs()) {
				throw new IOException("创建临时目录失败");
			}
		}

		return tmpDir;
	}

	private void reset() {
		apkProperty = null;
		apkIconResourceProperty = null;
		lockThumbnailResourceProperties = null;
		lockThumbLeftResourceProperties = null;
		lockThumbRightResourceProperties = null;
		apkResourceMap = null;
		apkStringValueMap = null;
		isIconResourceSaved = false;
		isLockThumbnailSaved = false;
		isLockThumbLeftSaved = false;
		isLockThumbRightSaved = false;
		tmpDir = null;
		thumbPics = null;
	}

	//是否是manifest xml文件
	private boolean isManifest(String name) {
		return name.equals(MANIFEST_XML);
	}

	//是否是锁屏 xml文件
	private boolean isLockXML(String name) {
		return name.equals(LOCK_XML);
	}

	//是否是类文件
	private boolean isClassDex(String name) {
		return name.equals(CLASSES_DEX);
	}

	//是否是资源文件
	private boolean isResource(String name) {
		return name.startsWith(RES_DRAWABLE);
	}
	
	//是否是全局主题 apk 两张图片
	private boolean isGloadThemePre1(String name) {
		return name.contains(RES_DREVIEW1);
	}
	//是否是全局主题 apk 两张图片
	private boolean isGloadThemePre2(String name) {
		return name.contains(RES_DREVIEW2);
	}

	//是否是资源加密文件
	private boolean isArsc(String name) {
		return name.equals(RESOURCES_ARSC);
	}

	
	/**
	 * 解析指定路径的apk文件，并将图标保存为指定路径
	 * @param file 文件
	 * @param iconPath 图标的路径，若为空则不保存图标
	 * @param thumbPics  截图路径
	 * @return
	 */
	public String [] parseThemeInnerAPk(File file, String[] thumbPics) throws Exception {
	
		ZipFile zipFile = new ZipFile(file);
		Enumeration<ZipEntry> entries = zipFile.getEntries();
		try {
			while (entries.hasMoreElements()) {
				ZipEntry ze = entries.nextElement();
				if(!ze.isDirectory()){
					String name = ze.getName();
					if(isGloadThemePre1(name)){//全局主题 预览图1
					
						String fileSuffix=name.substring(name.lastIndexOf("."));
						thumbPics[0]=new StringBuffer().append(thumbPics[0]).append(fileSuffix).toString();
						FileOperate.saveStreamToFile(thumbPics[0], zipFile.getInputStream(ze));
					
					}else if(isGloadThemePre2(name)){//全局主题 预览图2
						String fileSuffix=name.substring(name.lastIndexOf("."));
						thumbPics[1]=new StringBuffer().append(thumbPics[1]).append(fileSuffix).toString();
						FileOperate.saveStreamToFile(thumbPics[1], zipFile.getInputStream(ze));
					}
				}
			}
		} finally {
			zipFile.close();
		}
		return thumbPics;
	}
	
	
	public static void main(String[] args) {
		long start = System.currentTimeMillis();
		try {
			ApkParser ap = new ApkParser("D:\\tmp");
			File file = new File("D:\\My Documents\\桌面\\Project-release.apk");
			String picDir = "D:\\tmp\\";
			String iconPath = picDir + "icon.jpg";
			String[] picsPaths = new String[3];
			picsPaths[0] = picDir + "1";
			picsPaths[1] = picDir + "2";
			picsPaths=ap.parseThemeInnerAPk(file, picsPaths);
			ApkProperty apk = ap.getApkProperty();
//			LockProperty lock = ap.getLockProperty();
			if(apk!=null){
				for(int i=0;i<apk.getIconAddress().length(); i++){
					System.out.println(apk.getIconAddress());
				}
				
			}
			
//			System.out.println(null == lock ? "null" : lock.getDescription());
		} catch(Exception ex) {
			ex.printStackTrace();
		}

		long total = System.currentTimeMillis() - start;
		System.out.println("耗时：" + total);

//		ResConfigFlags flag = new ResConfigFlags((short)0, (short)0, new char[]{'z', 'h'}, new char[]{'C', 'N'},
//				ResConfigFlags.SCREENLAYOUT_LAYOUTDIR_ANY, ResConfigFlags.ORIENTATION_ANY, ResConfigFlags.TOUCHSCREEN_ANY,
//				ResConfigFlags.DENSITY_DEFAULT, ResConfigFlags.KEYBOARD_ANY,
//				ResConfigFlags.NAVIGATION_ANY, (byte)(ResConfigFlags.KEYSHIDDEN_ANY | ResConfigFlags.NAVHIDDEN_ANY),
//				(short)0, (short)0, (short)0,
//				(byte)(ResConfigFlags.SCREENLONG_ANY | ResConfigFlags.SCREENSIZE_ANY),
//				(byte)(ResConfigFlags.UI_MODE_TYPE_ANY | ResConfigFlags.UI_MODE_NIGHT_ANY),
//				(short)0, (short)0, (short)0, false);
//		System.out.println(flag.getQualifiers());
	}
}


