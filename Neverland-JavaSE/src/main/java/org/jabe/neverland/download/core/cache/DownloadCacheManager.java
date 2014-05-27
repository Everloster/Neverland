package org.jabe.neverland.download.core.cache;

import java.io.IOException;

import org.jabe.neverland.download.core.DownloadInfo;

public abstract class DownloadCacheManager {
	
	/**
	 * 读取下载缓存
	 * 
	 * @param cacheTask
	 */
	public abstract void readFromCache(final DownloadCacheTask cacheTask)  throws IOException;
	
	/**
	 * 更新下载缓存
	 * 
	 * @param cacheTask
	 */
	public abstract void saveToCache(final DownloadCacheTask cacheTask)  throws IOException;
	
	/**
	 * 下载是否有缓存
	 * 
	 * @param cacheTask
	 * @return
	 */
	public abstract boolean isInCache(final DownloadCacheTask cacheTask);
	
	/**
	 * 更新下载进度
	 * 
	 * @param sectionNo 段编号
	 * @param progress 新增的数量
	 * @param cacheTask 
	 */
	public abstract void updateSectionProgress(final byte[] bytes, final int sectionNo, final long progress, final DownloadCacheTask cacheTask) throws IOException;
	
	/**
	 * 获取下载进度
	 * 
	 * @param downloadInfo
	 * @return
	 */
	public abstract int getDownloadedPercent(final DownloadInfo downloadInfo);
	
	/**
	 * 下载是否完成
	 * 
	 * @param downloadInfo
	 * @return
	 */
	public abstract boolean isDownloadFinished(final DownloadInfo downloadInfo);
	
	/**
	 * 根据下载信息生成    下载缓冲文件  存放的绝对路径
	 * 
	 * @param downloadInfo
	 * @return
	 */
	public abstract String generateCacheSaveFullPath(final DownloadInfo downloadInfo);
	
	/**
	 * 根据下载信息生成    下载完成文件  存放的绝对路径
	 * 
	 * @param downloadInfo
	 * @return
	 */
	public abstract String generateFinishPath(final DownloadInfo downloadInfo);
	
	/**
	 * 下载完成后的一些处理
	 * 
	 * @param cacheTask
	 * @return
	 */
	public abstract boolean completeCacheTask(final DownloadCacheTask cacheTask);
	
	/**
	 * 清除缓存
	 * @param cacheTask
	 * @return
	 */
	public abstract void clearCache(final DownloadCacheTask cacheTask);
	
}
