/**
 * Copyright (C) 2013, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	liu jing qiang
 * Since	2013-8-2
 */
package com.nearme.freeupgrade.lib.pinterface;

import java.io.File;
import java.io.IOException;

import org.xmlpull.v1.XmlPullParserException;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.content.res.Resources;
import android.content.res.XmlResourceParser;
import android.view.LayoutInflater;
import android.view.View;

import com.nearme.freeupgrade.lib.AppWidgetProviderInfo;
import com.nearme.freeupgrade.lib.PluginContextWrapper;
import com.nearme.freeupgrade.lib.net.NetAccessListener;
import com.nearme.freeupgrade.lib.net.NetManager;
import com.nearme.freeupgrade.lib.util.Constants;
import com.nearme.freeupgrade.lib.util.InnerUtil;

import dalvik.system.DexClassLoader;

/**
 * 主要存放一些对外开放的接口
 * 
 * @Author liu jing qiang
 * @Since 2013-8-2
 */
public class PluginInterface {

	/**
	 * 初始化一些配置
	 * 
	 * @param projectRootDir
	 *            主项目的根目录
	 */
	public static void initPluginConfig(File projectRootDir) {
		Constants.MAIN_PROJECT_DIR_PATH = projectRootDir.getAbsolutePath();
	}

	/**
	 * 获取插件apk中指定名称的layout转化为view
	 * 
	 * @param paramActivity
	 * @param apkPath
	 * @param layoutName
	 * @return
	 */
	public static View getInflateLayoutView(Context paramActivity, String apkPath, String layoutName) {
		PackageInfo packageInfo = paramActivity.getPackageManager().getPackageArchiveInfo(apkPath,
				PackageManager.GET_RECEIVERS);
		Context pluginContext = getPluginContext(paramActivity, apkPath, packageInfo.packageName);
		int layoutId = pluginContext.getResources().getIdentifier(layoutName, "layout",
				packageInfo.packageName);
		View view = LayoutInflater.from(pluginContext).inflate(layoutId, null);
		return view;
	}

	/**
	 * 针对widget,由xml文件夹appwidget_provider读取initialLayout ID,并加载返回View
	 * 
	 * @param paramActivity
	 * @param apkPath
	 * @param xmlName
	 * @return
	 * @throws NameNotFoundException
	 * @throws XmlPullParserException
	 * @throws IOException
	 */
	public static View getWidgetViewByXml(Context paramActivity, String apkPath, String xmlName)
			throws NameNotFoundException, XmlPullParserException, IOException {
		PackageInfo packageInfo = paramActivity.getPackageManager().getPackageArchiveInfo(apkPath,
				PackageManager.GET_RECEIVERS);
		Context pluginContext = getPluginContext(paramActivity, apkPath, packageInfo.packageName);
		Resources res = pluginContext.getResources();
		AppWidgetProviderInfo info = getWidgetProviderInfo(xmlName, res, packageInfo.packageName);
		if (info != null) {
			View view = LayoutInflater.from(pluginContext).inflate(info.getInitialLayout(), null);
			return view;
		}
		return null;
	}

	/**
	 * 获取OPPO Widget xml文件夹下widget描述信息
	 * 
	 * @param paramActivity
	 * @param apkPath
	 * @param xmlName
	 * @return
	 * @throws NameNotFoundException
	 * @throws XmlPullParserException
	 * @throws IOException
	 */
	public static AppWidgetProviderInfo getWidgetProviderInfoByXml(Context paramActivity,
			String apkPath, String xmlName) throws NameNotFoundException, XmlPullParserException,
			IOException {
		PackageInfo packageInfo = paramActivity.getPackageManager().getPackageArchiveInfo(apkPath,
				PackageManager.GET_RECEIVERS);
		Context pluginContext = getPluginContext(paramActivity, apkPath, packageInfo.packageName);
		Resources res = pluginContext.getResources();
		return getWidgetProviderInfo(xmlName, res, packageInfo.packageName);
	}

	/**
	 * 根据外部插件apk路径,apk插件包名获取该Apk替代Context，由此可以获得该apk Resources信息
	 * 
	 * @param paramActivity
	 * @param apkPath
	 * @param pkgName
	 * @return
	 */
	public static Context getPluginContext(Context paramActivity, String apkPath, String pkgName) {
		DexClassLoader localDexClassLoader = InnerUtil.getDexClassLoaderByPath(paramActivity,
				apkPath, pkgName);
		return new PluginContextWrapper(paramActivity, 0, apkPath, localDexClassLoader);
	}

	private static AppWidgetProviderInfo getWidgetProviderInfo(String xmlName, Resources res,
			String packageName) throws NameNotFoundException, XmlPullParserException, IOException {
		int xmlId = res.getIdentifier(xmlName, "xml", packageName);
		XmlResourceParser parser = res.getXml(xmlId);
		return parserWidgetProviderInfo(parser);
	}

	private static AppWidgetProviderInfo parserWidgetProviderInfo(XmlResourceParser parser)
			throws XmlPullParserException, IOException, NameNotFoundException {
		AppWidgetProviderInfo info = null;
		try {
			if (parser == null) {
				return null;
			}
			int xmlEventType;
			while ((xmlEventType = parser.next()) != XmlResourceParser.END_DOCUMENT) {
				if (xmlEventType == XmlResourceParser.START_TAG) {
					String nameSpace = parser.getAttributeNamespace(0);
					int initialLayout = parser.getAttributeResourceValue(nameSpace,
							"initialLayout", 0);
					int minWidth = parser.getAttributeResourceValue(nameSpace, "minWidth", 0);
					int minHeight = parser.getAttributeResourceValue(nameSpace, "minHeight", 0);
					int updatePeriodMillis = parser.getAttributeResourceValue(nameSpace,
							"updatePeriodMillis", 0);
					int previewImage = parser.getAttributeResourceValue(nameSpace, "previewImage",
							0);
					int minResizeWidth = parser.getAttributeResourceValue(nameSpace,
							"minResizeWidth", 0);
					int minResizeHeight = parser.getAttributeResourceValue(nameSpace,
							"minResizeHeight", 0);
					info = new AppWidgetProviderInfo();
					info.setInitialLayout(initialLayout);
					info.setMinWidth(minWidth);
					info.setMinHeight(minHeight);
					info.setUpdatePeriodMillis(updatePeriodMillis);
					info.setPreviewImage(previewImage);
					info.setmMinResizeWidth(minResizeWidth);
					info.setMinResizeHeight(minResizeHeight);
				}
			}
		} catch (XmlPullParserException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (parser != null)
				parser.close();
		}
		return info;
	}
	
	/**
	 * 检查插件状态
	 * 
	 * @param context
	 * @param listener
	 */
	public static void checkPlugins(Context context, NetAccessListener listener) {
		NetManager.checkPluginList(context, listener, InnerUtil.getPluginDir());
	}
}
