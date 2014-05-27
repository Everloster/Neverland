package org.jabe.neverland.download.core.impl;

import java.util.concurrent.ExecutorService;

import org.jabe.neverland.download.core.DownloadEngine;
import org.jabe.neverland.download.core.DownloadOperationMaper;
import org.jabe.neverland.download.core.cache.DownloadCacheManager;
import org.jabe.neverland.download.core.cache.impl.FileCacheManager;
import org.jabe.neverland.download.core.engine.impl.DefaultOperationMaper;
import org.jabe.neverland.download.core.engine.impl.TaskDownloadEngine;

public class DefaultConfigurationFactory {

	public static DownloadEngine getDefaultDownloadEngine(DownloadCacheManager progressCacheManager, ExecutorService executorService) {
		return new TaskDownloadEngine(progressCacheManager, executorService);
	}
	
	public static DownloadOperationMaper getDefaultMaper(DownloadEngine downloadEngine) {
		return new DefaultOperationMaper(downloadEngine);
	}
	
	public static DownloadCacheManager getDefaultCacheManager(String rootPath) {
		return new FileCacheManager(rootPath);
	}
}
