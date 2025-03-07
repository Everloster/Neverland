package org.jabe.neverland.download.test;

import java.util.concurrent.CountDownLatch;

import org.jabe.neverland.download.core.DownloadStatus;
import org.jabe.neverland.download.core.DownloadStatusListener;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;


public class CancelContinueTest extends DownloadTestBase {
	
	@Before
	public void setUp() {
		super.setUp();
	}
	
	@Test
	public void testContinue() {
		final CountDownLatch latcher = new CountDownLatch(1);
		dm.registerListener(new DownloadStatusListener() {

			@Override
			public void onStatusCountChanged(int downloadingCount) {
				
			}

			@Override
			public void onStatusChanged(String packageName, int status) {
				if (status == DownloadStatus.DOWNLOAD_STATUS_FINISHED.ordinal()) {
					System.out
							.println(packageName + " has finish downloading.");
					Assert.assertTrue(true);
					latcher.countDown();
				}
			}

			@Override
			public void onFailure(String packageName, Exception e) {
				Assert.fail();
				latcher.countDown();
			}
		});
		
		dm.startDownload(downloadInfo);
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		dm.cancelDownload(downloadInfo);
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		dm.startDownload(downloadInfo);
		try {
			latcher.await();
		} catch (InterruptedException e1) {
			e1.printStackTrace();
			Assert.fail();
		}
	}
}
