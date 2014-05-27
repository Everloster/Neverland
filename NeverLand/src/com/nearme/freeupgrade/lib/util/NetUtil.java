package com.nearme.freeupgrade.lib.util;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URL;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.text.TextUtils;

/**
 * Title: CHttpUtil.java <br/>
 * Copyright: Copyright (c) 2013 www.oppo.com Inc. All rights reserved. <br/>
 * Company: OPPO <br/>
 * 
 * @author 80054358
 * @version 1.0
 * 
 *          网络相关的工具类
 */
public class NetUtil {
	private static final String CHINAUNICOMWAP = "uniwap";
	private static final String CHINAUNICOM3GNET = "3gwap";
	private static final String CHINAMOBILE_CMWAP = "cmwap";
	public static final int NO_NET = 0;
	public static final int WIFI = 1;
	public static final int MOBILE_GOOD = 3;// 3G网络
	public static final int MOBILE_BAD = 4;// 2G网络

	public static final String OPERATOR_CMCC = "CMCC";// 移动运营商
	public static final String OPERATOR_UNICOM = "UNICOM";// 联通运营商
	public static final String OPERATOR_TELECOM = "TELECOM";// 电信运营商

	/**
	 * 判断是否是移动wap网络，是true，否false
	 * 
	 * @param context
	 * @return
	 */
	public static boolean isChinaMobileWap(Context context) {
		if (null != context) {
			ConnectivityManager connectivityManager = (ConnectivityManager) context
					.getSystemService(Context.CONNECTIVITY_SERVICE);
			NetworkInfo networkInfo = connectivityManager.getActiveNetworkInfo();
			if (null != networkInfo) {
				if (networkInfo.isAvailable()
						&& ConnectivityManager.TYPE_WIFI != networkInfo.getType()) {// 是移动网络
					if (CHINAMOBILE_CMWAP.equals(networkInfo.getExtraInfo())) {
						return true;
					} else {
						return false;
					}
				}
			}
		}
		return false;
	}

	public static boolean isMobileNotChinaUniocomWap(Context context) {
		if (null != context) {
			ConnectivityManager connectivityManager = (ConnectivityManager) context
					.getSystemService(Context.CONNECTIVITY_SERVICE);
			NetworkInfo networkInfo = connectivityManager.getActiveNetworkInfo();
			if (null != networkInfo) {
				if (networkInfo.isAvailable()
						&& ConnectivityManager.TYPE_WIFI != networkInfo.getType()) {// 是移动网络
					if (CHINAUNICOMWAP.equals(networkInfo.getExtraInfo())
							|| CHINAUNICOM3GNET.equals(networkInfo.getExtraInfo())) {
						return false;
					} else {
						return true;
					}
				}
			}
		}
		return false;
	}

	/**
	 * 创建接口访问的HttpURLConnection
	 * 
	 * @param context
	 * @param urlStr
	 * @return
	 * @throws IOException
	 */
	public static HttpURLConnection createUrlConnecttion(Context context, String urlStr)
			throws IOException {
		URL url = new URL(urlStr);
		HttpURLConnection connection = null;

		// 移动wap时做特殊处理
		if (isChinaMobileWap(context)) {
			String newUrl = "http://10.0.0.172" + url.getPath();
			String domain = url.getHost();
			int port = url.getPort();
			URL url2 = new URL(newUrl);
			connection = (HttpURLConnection) url2.openConnection();
			if (port != -1) {
				domain = domain + ":" + port;
			}
			connection.setRequestProperty("X-Online-Host", domain);
		} else if (isMobileNotChinaUniocomWap(context)) {
			// String proxyHost = android.net.Proxy.getDefaultHost();
			// int proxyPort = android.net.Proxy.getDefaultPort();
			String proxyHost = getDefaultHost();
			int proxyPort = getDefaultPort();

			if (!TextUtils.isEmpty(proxyHost)) {
				java.net.Proxy proxy = new java.net.Proxy(java.net.Proxy.Type.HTTP,
						new InetSocketAddress(proxyHost, proxyPort));
				connection = (HttpURLConnection) url.openConnection(proxy);
			}
		}

		if (connection == null)
			connection = (HttpURLConnection) url.openConnection();
		connection.setConnectTimeout(30000);
		connection.setReadTimeout(30000);
		return connection;
	}

	/**
	 * 获取代理Host
	 * 
	 * @return
	 */
	public static final String getDefaultHost() {
		String host = System.getProperty("http.proxyHost");
		if (TextUtils.isEmpty(host))
			return null;
		return host;
	}

	/**
	 * 获取代理Port
	 * 
	 * @return
	 */
	public static final int getDefaultPort() {
		if (getDefaultHost() == null)
			return -1;
		try {
			return Integer.parseInt(System.getProperty("http.proxyPort"));
		} catch (NumberFormatException e) {
			return -1;
		}
	}

	public static String getWapType(Context context) {
		if (null != context) {
			ConnectivityManager connectivityManager = (ConnectivityManager) context
					.getSystemService(Context.CONNECTIVITY_SERVICE);
			NetworkInfo networkInfo = connectivityManager.getActiveNetworkInfo();
			if (null != networkInfo) {
				if (networkInfo.isAvailable()
						&& ConnectivityManager.TYPE_WIFI != networkInfo.getType()) {// 是移动网络
					return networkInfo.getExtraInfo();
				}
			}
		}
		return "";
	}

	/**
	 * 判断当前是否为wifi网络
	 * 
	 * @param ctx
	 * @return
	 */
	public static boolean isWifiWorking(Context ctx) {
		ConnectivityManager connectivityManager = (ConnectivityManager) ctx
				.getSystemService(Context.CONNECTIVITY_SERVICE);
		NetworkInfo activeNetInfo = connectivityManager.getActiveNetworkInfo();
		if (activeNetInfo != null && activeNetInfo.getType() == ConnectivityManager.TYPE_WIFI) {
			return true;
		}
		return false;
	}

	/**
	 * 判断是否有网络
	 * 
	 * @param ctx
	 * @return
	 */
	public static boolean isNetWorking(Context ctx) {
		ConnectivityManager connectivity = (ConnectivityManager) ctx
				.getSystemService(Context.CONNECTIVITY_SERVICE);
		if (connectivity != null) {
			NetworkInfo[] info = connectivity.getAllNetworkInfo();
			if (info != null) {
				for (int i = 0; i < info.length; i++) {
					if (info[i].getState() == NetworkInfo.State.CONNECTED) {
						return true;
					}
				}
			}
			if (connectivity.getActiveNetworkInfo() != null
					&& connectivity.getActiveNetworkInfo().isAvailable()) {
				return true;
			}
		}
		return false;
	}
}
