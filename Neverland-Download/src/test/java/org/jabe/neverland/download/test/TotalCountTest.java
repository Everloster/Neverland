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
public class TotalCountTest extends DownloadTestBase {

	/**
	 * 
	 */
	public TotalCountTest() {
	}
	
	@Before
	public void setUp() {
		super.setUp();
	}
	
	
	@Test
	public void testTotalCount() {
		final CountDownLatch latcher = new CountDownLatch(2);
		dm.registerListener(new DownloadStatusListener() {

			@Override
			public void onStatusCountChanged(int downloadingCount) {
				System.out.println("current downloading count : " + downloadingCount);
				if (downloadingCount == 1) {
					latcher.countDown();
				}
				if (downloadingCount == 0) {
					latcher.countDown();
					Assert.assertTrue(true);
				}
			}

			@Override
			public void onStatusChanged(String packageName, int status) {
				if (status == DownloadStatus.DOWNLOAD_STATUS_FINISHED.ordinal()) {
					System.out
							.println(packageName + " has finish downloading.");
				}
			}

			@Override
			public void onFailure(String packageName, Exception e) {
				Assert.fail();
			}
		});
		dm.startDownload(downloadInfo);
		try {
			latcher.await();
		} catch (InterruptedException e1) {
			e1.printStackTrace();
			Assert.fail();
		}
	}
	

}
