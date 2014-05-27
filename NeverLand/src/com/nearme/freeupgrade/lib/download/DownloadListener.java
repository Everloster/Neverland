/**
 * 
 */
package com.nearme.freeupgrade.lib.download;

import com.nearme.freeupgrade.lib.db.PluginInfo;

/**
 * @author 80049063
 * 
 */
public interface DownloadListener {
	public void onDownloadFail(String pkgName);

	public void onDownloadFatal(String pkgName);

	public void onDownloadStart(String pkgName);

	public void onDownloadSuccess(PluginInfo info);

	public void onDownloadUpdate(String pkgName);
}
