package com.nearme.freeupgrade.lib.net;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;

import org.apache.http.HttpEntity;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.text.TextUtils;

import com.nearme.freeupgrade.lib.pinterface.PluginItem;
import com.nearme.freeupgrade.lib.util.Constants;
import com.nearme.freeupgrade.lib.util.InnerUtil;
import com.nearme.freeupgrade.lib.util.LogUtil;
import com.nearme.freeupgrade.lib.util.NetUtil;

/**
 * Title: NetAccess.java <br/>
 * Copyright: Copyright (c) 2013 www.oppo.com Inc. All rights reserved. <br/>
 * Company: OPPO <br/>
 * 
 * @author 80054358
 * @version 1.0
 * 
 *          网络访问
 */
public class NetManager {

	public static final int OP_CHECK_PLUGIN_LIST = 1001;

	public static class ConnectTask extends AsyncTask<Void, Void, HttpResult> {
		Context mContext;
		String mUrl;
		NetAccessListener mNetAccessListener;
		int opCode;

		public ConnectTask(Context context, int opCode, NetAccessListener listener) {
			this.mContext = context;
			this.mNetAccessListener = listener;
			this.opCode = opCode;
		}

		protected String getUrl() {
			return null;
		}

		protected String getRequestBody() {
			return null;
		}

		protected boolean onFinished(Object obj) throws Exception {
			return false;
		}

		protected void onFail(int statusCode, String errorMsg) throws Exception {
			mNetAccessListener.onNetAccessFail(opCode, errorMsg);
		}

		/**
		 * 返回返回体所解析出的对象
		 * 
		 * @param response
		 * @return
		 * @throws Exception
		 */
		protected Object getReturnObject(Object dataBody) throws Exception {
			return null;
		}

		byte[] getGZipResponseBody(InputStream inputStream) {
			try {
				ByteArrayOutputStream baos = new ByteArrayOutputStream();
				byte[] buf = new byte[2048];
				GZIPInputStream is = new GZIPInputStream(inputStream);
				for (int i = 0; i != -1; i = is.read(buf))
					baos.write(buf, 0, i);
				LogUtil.i(Constants.TAG, "GZipContentLength=:" + baos.toByteArray().length);
				return baos.toByteArray();
			} catch (Exception e) {
				return null;
			}
		}

		/**
		 * 返回protocol buffer返回体
		 * 
		 * @param entity
		 * @return
		 */
		byte[] getResponseBody(HttpEntity entity) {
			try {
				ByteArrayOutputStream baos = new ByteArrayOutputStream();
				byte[] buf = new byte[2048];
				InputStream is = entity.getContent();
				for (int i = 0; i != -1; i = is.read(buf))
					baos.write(buf, 0, i);
				LogUtil.i(Constants.TAG, "ContentLength=:" + baos.toByteArray().length);
				return baos.toByteArray();
			} catch (Exception e) {
				return null;
			}
		}

		/**
		 * 返回protocol buffer返回体
		 * 
		 * @param entity
		 * @return
		 */
		byte[] getResponseBody(InputStream inputStream) {
			try {
				ByteArrayOutputStream baos = new ByteArrayOutputStream();
				byte[] buf = new byte[2048];
				for (int i = 0; i != -1; i = inputStream.read(buf))
					baos.write(buf, 0, i);
				LogUtil.i(Constants.TAG, "ContentLength=:" + baos.toByteArray().length);
				return baos.toByteArray();
			} catch (Exception e) {
				return null;
			}
		}

		/**
		 * 返回xml返回体
		 * 
		 * @param entity
		 * @param isGZip
		 * @return
		 */
		String getResponseBodyAsString(InputStream inputStream, boolean isGZip) {
			if (isGZip) {
				return InnerUtil.getUTF8String(getGZipResponseBody(inputStream));
			} else {
				return InnerUtil.getUTF8String(getResponseBody(inputStream));
			}
		}

		@Override
		protected HttpResult doInBackground(Void... params) {
			try {
				HttpResult result = new HttpResult();
				LogUtil.i(Constants.TAG, " Url: " + getUrl());
				HttpURLConnection conn = NetUtil.createUrlConnecttion(mContext, getUrl());
				conn.setReadTimeout(30000);
				conn.setConnectTimeout(30000);
				// xml形式
				conn.setRequestProperty("Content-Type", "text/xml; charset=UTF-8");
				// conn.setRequestProperty("Accept-Encoding", "gzip");
				String requesBody = getRequestBody();
				LogUtil.i(Constants.TAG, " requestBody: " + requesBody);
				if (!TextUtils.isEmpty(requesBody)) {
					conn.setDoOutput(true);
					conn.setRequestMethod("POST");
					OutputStream outStream = conn.getOutputStream();
					outStream.write(InnerUtil.getUTF8Bytes(requesBody));
				} else {
					conn.setRequestMethod("GET");
				}
				conn.connect();
				// String body = (String)conn.getContent();
				// xml协议
				result.responseBody = getResponseBodyAsString(conn.getInputStream(), false);
				result.statusCode = conn.getResponseCode();
				result.returnObject = getReturnObject(result.responseBody);
				LogUtil.i(Constants.TAG, " statusCode: " + result.statusCode);
				LogUtil.i(Constants.TAG, " responseBody: " + result.responseBody);
				return result;
			} catch (Exception e) {
				e.printStackTrace();
			}
			return null;
		}

		@Override
		protected void onPostExecute(HttpResult result) {
			try {
				if (result == null) {
					onFail(0, null);
				} else {
					if (result.statusCode == 200) {
						onFinished(result.returnObject);
						mNetAccessListener.onNetAccessSuccess(opCode, result.returnObject);
					} else {
						onFail(result.statusCode, null);
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			super.onPostExecute(result);
		}
	}

	/*********************************************** 以下是具体接口实现 *********************************************************/

	/**
	 * 检查插件列表信息
	 */
	public static void checkPluginList(final Context context, NetAccessListener listener,
			final File pluginDir) {
		new ConnectTask(context, OP_CHECK_PLUGIN_LIST, listener) {
			protected String getUrl() {
				String url = "http://";
				return url;
			};
			
			protected String getRequestBody() {
				//解析路径下的插件，并删除数据库中的冗余数据
				ArrayList<PluginItem> pluginList = new ArrayList<PluginItem>();
				try {
					File[] pluginFiles = pluginDir.listFiles();
					for (int i = 0; i < pluginFiles.length; i++) {
						if (!pluginFiles[i].getAbsolutePath().endsWith(Constants.SUFFIX_PLUGIN)) {
							continue;
						}
						PluginItem tempItem = new PluginItem();
						File tempFile = pluginFiles[i];
						PackageInfo packageInfo = context.getPackageManager()
								.getPackageArchiveInfo(tempFile.getAbsolutePath(), 0);
//						tempItem.pid = packageInfo.applicationInfo.metaData.get("plugin_id")
//								.toString();
						tempItem.pkgName = packageInfo.packageName;
						tempItem.versionCode = packageInfo.versionCode;
						tempItem.fileMd5 = InnerUtil.getFileMD5(tempFile);
						pluginList.add(tempItem);
					}
				} catch (Exception e) {
					e.printStackTrace();
				}
				//组装xml
				StringBuffer buffer = new StringBuffer();
				buffer.append("<request local_version=\"1\"");
				buffer.append(" main_package_name=\""+context.getPackageName()+"\"");
				buffer.append(" sdk_version=\""+InnerUtil.getSdkApi()+"\"");
				buffer.append(" screen_size=\""+InnerUtil.getScreenSize(context)+"\"");
				buffer.append(" mobile_name=\""+InnerUtil.getMobileName()+"\"");
				buffer.append(" imei=\""+InnerUtil.getIMEI(context)+"\">");
				for (int i = 0; i < pluginList.size(); i++) {
					PluginItem tempItem = pluginList.get(i);
					buffer.append("<plugin id=\""+tempItem.pid+"\"");
					buffer.append(" pakcage_name=\""+tempItem.pkgName+"\"");
					buffer.append(" version_code=\""+tempItem.versionCode+"\"");
					buffer.append(" md5=\""+tempItem.fileMd5+"\"/>");
				}
				buffer.append("</request>");
				return buffer.toString();
			};

			@Override
			protected Object getReturnObject(Object dataBody) throws Exception {
				return XMLParser.parsePlugins((String) dataBody);
			}

		}.execute();
	}

}
