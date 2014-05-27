/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年4月3日
 */
package org.jabe.neverland.download.test;

import java.util.concurrent.CountDownLatch;

import org.jabe.neverland.download.core.DownloadStatus;
import org.jabe.neverland.download.core.DownloadStatusListener;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * 
 * @Author	LaiLong
 * @Since	2014年4月3日
 */
public class WaitTaskTest extends DownloadTestBase {

	/**
	 * 
	 */
	public WaitTaskTest() {
	}
	
	@Before
	public void setUp() {
		super.setUp();
	}
	
	@Test
	public void testWaitTask() {
		final CountDownLatch latcher = new CountDownLatch(4);
		dm.registerListener(new DownloadStatusListener() {

			@Override
			public void onStatusCountChanged(int downloadingCount) {
				
			}

			@Override
			public void onStatusChanged(String packageName, int status) {
				if (status == DownloadStatus.DOWNLOAD_STATUS_FINISHED.ordinal()) {
					System.out
							.println(packageName + " has finish downloading.");
					latcher.countDown();
				}
				
				if (status == DownloadStatus.DOWNLOAD_STATUS_WAIT.ordinal()) {
					System.out
							.println(packageName + " has been added to waiting queue.");
					latcher.countDown();
				}
			}

			@Override
			public void onFailure(String packageName, Exception e) {
				Assert.fail();
			}
		});
		
		dm.startDownload(downloadInfo);
		dm.startDownload(downloadInfo2);
		dm.startDownload(downloadInfo3);
		try {
			latcher.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
			Assert.fail();
		}
	}

}
