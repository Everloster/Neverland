package com.nearme.freeupgrade.lib.download;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.net.UnknownHostException;

import org.apache.http.conn.ConnectTimeoutException;

import android.content.Context;
import android.os.Process;
import android.text.TextUtils;

import com.nearme.freeupgrade.lib.db.PluginInfo;
import com.nearme.freeupgrade.lib.util.Constants;
import com.nearme.freeupgrade.lib.util.InnerUtil;
import com.nearme.freeupgrade.lib.util.LogUtil;
import com.nearme.freeupgrade.lib.util.NetUtil;

public class DownloadThread extends Thread {

	boolean bFirst = true; // 是否第一次取文件
	boolean bStop = false; // 停止标志
	private FileAccessI fileAccessI = null;
	private long flagPos;
	private double flagProgress;
	private DownloadListener mListener;
	DataOutputStream output; // 输出到文件的输出流
	private int retryTime = 0;
	private int mNewVersionCode = 0;
	PluginInfo siteInfoBean = null; // 文件信息Bean
	// byte[] key = DigestUtils.md5("oppo_comoppo_com");
	// Crypter sCrypter = new Crypter();
	private Context mContext;
	private boolean confirmNet = false;
	private boolean lastUseProxy = true;
	private boolean startFromZero = false;// 标志是否从0开始下载的
	private long startTime = 0;// 下载开始的时间
	private long endTime = 0;// 下载结束的时间
	private int BACK_SIZE = 1024;// 用于下载中断，继续下载时回退的字节数
	private int BUFFER_SIZE = 4096;// 下载的buffer大小
	private URL mUrl;

	public DownloadThread(Context ctx, PluginInfo bean, DownloadListener listener,
			int newVersionCode, boolean needDownloadPreparation) {
		mContext = ctx;
		siteInfoBean = bean;
		mListener = listener;
		mNewVersionCode = newVersionCode;
	}

	public void broacastDowloadUpdate() {
		if (mListener != null)
			mListener.onDownloadUpdate(siteInfoBean.pkgName);
	}

	public void broadcastDownloadFail(String errorMsg) {
		if (mListener != null)
			mListener.onDownloadFail(siteInfoBean.pkgName);
	}

	public void broadcastDownloadFatal(String errorMsg) {
		siteInfoBean.currentSize = 0;
		InnerUtil.getDownloadTempFile(siteInfoBean.fileName).delete();
		if (mListener != null)
			mListener.onDownloadFatal(siteInfoBean.pkgName);
	}

	public void broadcastDownloadStart() {
		if (mListener != null)
			mListener.onDownloadStart(siteInfoBean.pkgName);
	}

	public void broadcastDownloadSuccess(long time) {
		if (mListener != null) {
			mListener.onDownloadSuccess(siteInfoBean);
		}
	}

	public PluginInfo getDownloadInfo() {
		PluginInfo info = new PluginInfo();
		synchronized (siteInfoBean) {
			info = (PluginInfo) siteInfoBean.clone();
		}
		return info;
	}

	/**
	 * 获取主下载文件
	 * 
	 * @param info
	 * @return
	 */
	private File getMainfile(PluginInfo info) {
		File mainFile = InnerUtil.getDownloadTempFile(info.fileName);
		return mainFile;
	}

	public void run() {
		Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND);
		siteInfoBean.status = Constants.STATUS_DOWNLOADING;
		broadcastDownloadStart();

		File mainFile = getMainfile(siteInfoBean);

		// 标示是否从0开始下载
		if (siteInfoBean.currentSize == 0) {
			startFromZero = true;
			startTime = System.currentTimeMillis();
		} else if (siteInfoBean.currentSize != siteInfoBean.fileSize) {
			// 不是从0开始的回退BACK_SIZE个字节，用于减少丢包情况
			siteInfoBean.currentSize = siteInfoBean.currentSize - BACK_SIZE > 0 ? siteInfoBean.currentSize
					- BACK_SIZE
					: 0;
		}

		HttpURLConnection httpConnection = null;
		HttpURLConnection mainConnection = null;
		/**
		 * 下载过程
		 */
		while (true && !bStop) {
			try {
				// 如果版本改变则清空所有的
				if (mNewVersionCode != -1 && siteInfoBean.versionCode != -1
						&& siteInfoBean.versionCode != mNewVersionCode) {
					LogUtil.i(Constants.TAG, "version not equal:" + siteInfoBean.versionCode + ":"
							+ mNewVersionCode);
					mainFile.delete();
					siteInfoBean.versionCode = mNewVersionCode;
				}
				if (siteInfoBean.versionCode == -1) {
					siteInfoBean.versionCode = mNewVersionCode;
				}

				LogUtil.i(Constants.TAG,
						"mainfile:" + siteInfoBean.currentSize + ":" + mainFile.length());
				// 主文件不存在或者
				if (!mainFile.exists()) {
					LogUtil.i(Constants.TAG, "restart mainfile");
					mainFile.delete();
					mainFile.createNewFile();
					siteInfoBean.currentSize = 0;
				}

				if (siteInfoBean.currentSize > siteInfoBean.fileSize) {
					LogUtil.i(Constants.TAG, "mainfile length fatal:" + siteInfoBean.currentSize
							+ ":" + siteInfoBean.fileSize);
					broadcastDownloadFatal("DOWNLOAD_FIAL_CURRENT_MORE_FILE");
					return;
				} else if (siteInfoBean.currentSize == 0
						|| siteInfoBean.currentSize < siteInfoBean.fileSize) {
					String deUrl = siteInfoBean.fileUrl;
					// if (!deUrl.startsWith("http://")) {
					// byte[] result =
					// Base64.decodeBase64(Util.getUTF8Bytes(deUrl));
					// deUrl = Util.getUTF8String(sCrypter.decrypt(result,
					// key));
					// }
					LogUtil.i(Constants.TAG, "mainfile download:" + siteInfoBean.currentSize + ":"
							+ siteInfoBean.fileSize + ":" + deUrl);
					mUrl = new URL(deUrl);
					mainConnection = getUrlConnecttion(mUrl);

					String sProperty = "bytes=" + siteInfoBean.currentSize + "-";
					mainConnection.setRequestProperty("RANGE", sProperty);
					mainConnection.setDoInput(true);
					mainConnection.connect();
					LogUtil.i(
							Constants.TAG,
							"content-length:" + mainConnection.getContentLength() + ":"
									+ mainConnection.getResponseCode() + ":"
									+ mainConnection.getResponseMessage());
					if (mainConnection.getContentLength() == 0) {
						LogUtil.i(Constants.TAG, "mainfile content-length=0");
						siteInfoBean.status = Constants.STATUS_PAUSE;
						broadcastDownloadFatal("DOWNLOAD_FAIL_CONTENTLENGHT_EQUAL_0");
						return;
					} else if (mainConnection.getContentLength() < 0) {
						LogUtil.i(Constants.TAG,
								"mainfile content-length<0:" + mainConnection.getContentLength()
										+ ":" + mainConnection.getResponseCode() + ":"
										+ mainConnection.getResponseMessage());
						if (retryTime++ > 30) {
							siteInfoBean.status = Constants.STATUS_PAUSE;
							broadcastDownloadFail("DOWNLOAD_FAIL_CONTENTLENGHT_LESS_0");
							return;
						}
						try {
							Thread.sleep(1000);
						} catch (Exception e) {
						}
						continue;
					}

					if (siteInfoBean.currentSize == 0) {
						siteInfoBean.fileSize = mainConnection.getContentLength()
								+ siteInfoBean.currentSize;
						LogUtil.i(Constants.TAG, "siteInfoBean.currentSize == 0:"
								+ siteInfoBean.fileSize);
					} else if (mainConnection.getContentLength() + siteInfoBean.currentSize != siteInfoBean.fileSize) {
						LogUtil.i(Constants.TAG, mainConnection.getContentLength()
								+ siteInfoBean.currentSize + "!=" + siteInfoBean.fileSize);
						siteInfoBean.status = Constants.STATUS_PAUSE;
						siteInfoBean.currentSize = 0;
						siteInfoBean.fileSize = 0;
						broadcastDownloadFail("DOWNLOAD_FAIL_SIZE_NOT_EQUAL");
						return;
					}

					InputStream input = mainConnection.getInputStream();
					byte[] b = new byte[BUFFER_SIZE];
					int nRead;
					flagProgress = (double) siteInfoBean.currentSize
							/ (double) siteInfoBean.fileSize;
					flagPos = siteInfoBean.currentSize;
					fileAccessI = new FileAccessI(mainFile.getAbsolutePath(),
							siteInfoBean.currentSize);
					while ((nRead = input.read(b, 0, BUFFER_SIZE)) > 0 && !bStop) {
						confirmNet = true;
						siteInfoBean.currentSize += fileAccessI.write(b, 0, nRead);
						double currentProgress = (double) siteInfoBean.currentSize
								/ (double) siteInfoBean.fileSize;
						if (currentProgress - flagProgress > 0.02
								&& siteInfoBean.currentSize - flagPos > 50 * 1024) {
							LogUtil.i(Constants.TAG, "progress=" + (currentProgress - flagProgress)
									+ ":" + lastUseProxy);
							flagProgress = currentProgress;
							flagPos = siteInfoBean.currentSize;
							broacastDowloadUpdate();
						}
					}
					input.close();
					broacastDowloadUpdate();
				}

				/**
				 * 文件下载完成后处理方式：
				 */
				if (siteInfoBean.currentSize == siteInfoBean.fileSize) {
					siteInfoBean.status = Constants.STATUS_DOWNLOADED;
					// 如果是从0开始下载的，统计下载速度
					if (startFromZero) {
						endTime = System.currentTimeMillis();
						long time = endTime - startTime;
						broadcastDownloadSuccess(time);
					} else {
						broadcastDownloadSuccess(-1);
					}
					return;
				} else if (siteInfoBean.currentSize > siteInfoBean.fileSize) {
					LogUtil.i(Constants.TAG, "download finish length fial:"
							+ siteInfoBean.currentSize + ":" + siteInfoBean.fileSize);
					siteInfoBean.status = Constants.STATUS_PAUSE;
					broadcastDownloadFatal("DOWNLOAD_FIAL_CURRENT_MORE_FILE");
					return;
				}

			} catch (FileNotFoundException ex) {
				ex.printStackTrace();
				if (siteInfoBean.currentSize == siteInfoBean.fileSize) {
					siteInfoBean.status = Constants.STATUS_DOWNLOADED;
					// 如果是从0开始下载的，统计下载速度
					if (startFromZero) {
						endTime = System.currentTimeMillis();
						long time = endTime - startTime;
						broadcastDownloadSuccess(time);
					} else {
						broadcastDownloadSuccess(-1);
					}
					return;
				} else {
					siteInfoBean.status = Constants.STATUS_PAUSE;
					broadcastDownloadFail(ex.toString());
					return;
				}
			} catch (SocketTimeoutException ex1) {
				if (retryTime++ > 30) {
					ex1.printStackTrace();
					siteInfoBean.status = Constants.STATUS_PAUSE;
					broadcastDownloadFail(ex1.toString());
					return;
				}
			} catch (SocketException ex3) {
				if (retryTime++ > 30) {
					ex3.printStackTrace();
					siteInfoBean.status = Constants.STATUS_PAUSE;
					broadcastDownloadFail(ex3.toString());
					return;
				}
			} catch (ConnectTimeoutException ex2) {
				if (retryTime++ > 30) {
					ex2.printStackTrace();
					siteInfoBean.status = Constants.STATUS_PAUSE;
					broadcastDownloadFail(ex2.toString());
					return;
				}
			} catch (UnknownHostException ex2) {
				if (retryTime++ > 30) {
					ex2.printStackTrace();
					siteInfoBean.status = Constants.STATUS_PAUSE;
					broadcastDownloadFail(ex2.toString());
					return;
				}
			} catch (Exception e) {
				if (retryTime++ > 30) {
					e.printStackTrace();
					siteInfoBean.status = Constants.STATUS_PAUSE;
					broadcastDownloadFail(e.toString());
					return;
				}
			} finally {
				try {
					if (httpConnection != null) {
						httpConnection.getInputStream().close();
						httpConnection.disconnect();
					}
				} catch (Exception e) {
				}
				try {
					if (mainConnection != null) {
						mainConnection.getInputStream().close();
						mainConnection.disconnect();
					}
				} catch (Exception e) {
				}
			}
		}
	}

	public HttpURLConnection getUrlConnecttion(URL url) throws IOException {
		HttpURLConnection connection = null;
		LogUtil.i(Constants.TAG,
				NetUtil.getWapType(mContext) + ":" + android.net.Proxy.getDefaultHost() + ":"
						+ android.net.Proxy.getDefaultPort() + ":" + confirmNet + ":"
						+ lastUseProxy);
		if (NetUtil.isChinaMobileWap(mContext)) {
			String newUrl = "http://10.0.0.172" + url.getPath();
			String domain = url.getHost();
			int port = url.getPort();
			URL url2 = new URL(newUrl);
			connection = (HttpURLConnection) url2.openConnection();
			if (port != -1) {
				domain = domain + ":" + port;
			}
			connection.setConnectTimeout(30000);
			connection.setReadTimeout(30000);
			connection.setRequestProperty("X-Online-Host", domain);
		} else if (NetUtil.isMobileNotChinaUniocomWap(mContext)) {
			String proxyHost = android.net.Proxy.getDefaultHost();
			int proxyPort = android.net.Proxy.getDefaultPort();
			if (!TextUtils.isEmpty(proxyHost)) {
				java.net.Proxy proxy = new java.net.Proxy(java.net.Proxy.Type.HTTP,
						new InetSocketAddress(proxyHost, proxyPort));
				connection = (HttpURLConnection) url.openConnection(proxy);
				LogUtil.i(Constants.TAG, "set proxy");
				connection.setConnectTimeout(30000);
				connection.setReadTimeout(30000);
			}
		} else if (!NetUtil.isWifiWorking(mContext)) {
			boolean useProxy = false;
			if (confirmNet) {
				useProxy = lastUseProxy;
			} else {
				useProxy = !lastUseProxy;
			}
			if (useProxy) {
				String proxyHost = android.net.Proxy.getDefaultHost();
				int proxyPort = android.net.Proxy.getDefaultPort();
				if (!TextUtils.isEmpty(proxyHost)) {
					java.net.Proxy proxy = new java.net.Proxy(java.net.Proxy.Type.HTTP,
							new InetSocketAddress(proxyHost, proxyPort));
					connection = (HttpURLConnection) url.openConnection(proxy);
					connection.setConnectTimeout(30000);
					connection.setReadTimeout(30000);
					LogUtil.i(Constants.TAG, "set proxy");
				}
			}
			lastUseProxy = useProxy;
		}
		if (connection == null) {
			connection = (HttpURLConnection) url.openConnection();
			connection.setConnectTimeout(15000);
			connection.setReadTimeout(15000);
		}
		return connection;
	}

	public void stopDownload() {
		bStop = true;
	}
}
