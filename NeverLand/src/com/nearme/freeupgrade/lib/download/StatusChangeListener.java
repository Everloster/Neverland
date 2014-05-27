/**
 * 
 */
package com.nearme.freeupgrade.lib.download;

/**
 * @author 80054358
 * 
 */
public interface StatusChangeListener {

	/**
	 * 
	 * @param id
	 *            插件id
	 * @param action
	 *            行为状态码
	 * @param msgCode
	 *            附加码，比如下载失败原因
	 * @param msg
	 *            附加信息
	 */
	public void onStatusChange(String pkgName, int action, int msgCode, String msg);

}
