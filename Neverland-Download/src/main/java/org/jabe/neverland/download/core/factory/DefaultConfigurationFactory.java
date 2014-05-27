package org.jabe.neverland.download.core.factory;

import java.util.concurrent.ExecutorService;

import org.jabe.neverland.download.core.AbstractMessageDeliver;
import org.jabe.neverland.download.core.DownloadCacheManager;
import org.jabe.neverland.download.core.DownloadEngine;
import org.jabe.neverland.download.core.DownloadOperationMaper;
import org.jabe.neverland.download.core.cache.FileCacheManager;
import org.jabe.neverland.download.core.engine.CacheDownloadEngine;
import org.jabe.neverland.download.core.engine.DefaultOperationMaper;

public class DefaultConfigurationFactory {

	public static DownloadEngine getDefaultDownloadEngine(DownloadCacheManager progressCacheManager, ExecutorService executorService, AbstractMessageDeliver messageDeliver, int coreCount) {
		return new CacheDownloadEngine(progressCacheManager, executorService, messageDeliver, coreCount);
	}
	
	public static DownloadOperationMaper getDefaultMaper(DownloadEngine downloadEngine) {
		return new DefaultOperationMaper(downloadEngine);
	}
	
	public static DownloadCacheManager getDefaultCacheManager(String rootPath) {
		return new FileCacheManager(rootPath);
	}
}
