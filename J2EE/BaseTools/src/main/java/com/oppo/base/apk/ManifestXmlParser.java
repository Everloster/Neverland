/**
 * ManifestXmlParser.java
 * com.oppo.base.apk
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-7-16 		80036381
 *
 * Copyright (c) 2011 OPPO, All Rights Reserved.
*/

package com.oppo.base.apk;

import com.oppo.base.apk.entity.ApkProperty;

/**
 * ClassName:ManifestXmlParser
 * Function: AndroidManifest.xml文件解析
 *
 *
 * 调用方式：
 * 	ManifestXmlParser parser = new ManifestXmlParser();
 *  parser.parse("/var/resource/xml/AndroidManifest.xml");
 *  String xmlString = parser.getXmlString();
 *  ApkProperty apkProperty = parser.getApkProperty();
 * xmlString为解析后的xml字符串
 * apkProperty为解析后得到的manifest的基本属性
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-7-16  上午09:56:31
 *
 */
public class ManifestXmlParser extends ApkXmlParser {
	private static final String ANDROID_TAG_MANIFEST = "manifest";
	private static final String ANDROID_VERSION_CODE = "versionCode";
	private static final String ANDROID_VERSION_NAME = "versionName";
	private static final String ANDROID_PACKAGE = "package";

	private static final String ANDROID_TAG_APPLICATION = "application";
	private static final String ANDROID_ICON = "icon";
	private static final String ANDROID_LABEL = "label";

	private static final String ANDROID_TAG_USES_SDK = "uses-sdk";
	private static final String ANDROID_MIN_SDK = "minSdkVersion";

	private static final String ANDROID_TAG_USES_PERMISSION = "uses-permission";
	private static final String ANDROID_TAG_ACTIVITY = "activity";
	private static final String ANDROID_NAME = "name";

	private static final String PACKAGE_NAME_GAMECENTER = "com.nearme.gamecenter.open.core.LoginActivity";

	private ApkProperty apkProperty;
	private String tagName;

	public ManifestXmlParser() {
		super();
	}

	@Override
	public void reset() {
		super.reset();

		apkProperty = new ApkProperty();
	}

	public ApkProperty getApkProperty() {
		return apkProperty;
	}

	@Override
	protected void appendStartTag(String format, String indent, String tagName) {
		super.appendStartTag(format, indent, tagName);
		this.tagName = tagName;
	}

	@Override
	protected void appendEndTag(String format, String indent, String namespace, String name) {
		super.appendEndTag(format, indent, namespace, name);

		this.tagName = null;
	}

	@Override
	protected void appendNode(String format, String indent, String namespace, String nodeName, String value) {
		super.appendNode(format, indent, namespace, nodeName, value);

		if(ANDROID_TAG_MANIFEST.equals(tagName)) {
			if(ANDROID_VERSION_CODE.equals(nodeName)) {			//版本号
				apkProperty.setVersionCode(Integer.parseInt(value));
			} else if(ANDROID_VERSION_NAME.equals(nodeName)) {	//版本名
				apkProperty.setVersionName(value);
			} else if(ANDROID_PACKAGE.equals(nodeName)) {		//包名
				apkProperty.setPackageName(value);
			}
		} else if(ANDROID_TAG_USES_PERMISSION.equals(tagName)) {
			if(ANDROID_NAME.equals(nodeName)) {					//权限
				apkProperty.addPermission(value);
			}
		} else if(ANDROID_TAG_APPLICATION.equals(tagName)) {
			if(ANDROID_ICON.equals(nodeName)) {					//icon
				apkProperty.setIconAddress(value);
			} else if(ANDROID_LABEL.equals(nodeName)) {			//标签
				apkProperty.setLabel(value);
			}
		} else if(ANDROID_TAG_USES_SDK.equals(tagName)) {
			if(ANDROID_MIN_SDK.equals(nodeName)) {				//最低支持版本
				apkProperty.setMinSdkVersion(Integer.parseInt(value));
			}
		} else if(ANDROID_TAG_ACTIVITY.equals(tagName)) {
			if(ANDROID_NAME.equals(nodeName)) {
				if(PACKAGE_NAME_GAMECENTER.equals(value)) {
					apkProperty.setFromGamecenter(true);
				}
			}
		}
	}

	public static void main(String[] args) {
		ManifestXmlParser parser = new ManifestXmlParser();
		try {
			parser.parse("G:\\应用\\AndroidManifest.xml");
			System.out.println(parser.getXmlString());
			System.out.println(parser.getApkProperty().toString());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}

