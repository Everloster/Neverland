/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年4月8日
 */
package org.jabe.neverland.download.test;

import java.util.concurrent.CountDownLatch;

import org.jabe.neverland.download.core.DownloadInfo;
import org.jabe.neverland.download.core.DownloadProgressListener;
import org.jabe.neverland.download.core.DownloadStatus;
import org.jabe.neverland.download.core.DownloadStatusListener;
import org.jabe.neverland.download.util.SpeedDetectUtil;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

/**
 * 
 * @Author LaiLong
 * @Since 2014年4月8日
 */
public class SpeedTest extends DownloadTestBase {

	/**
	 * 
	 */
	public SpeedTest() {
	}

	@Before
	public void setUp() {
		super.setUp();
	}

	volatile long lastTime = 0;

	@Test
	public void testSpeed() {
		
		final CountDownLatch latcher = new CountDownLatch(1);
		final long startTime = System.currentTimeMillis();
		dm.registerListener(new DownloadProgressListener() {

			@Override
			public void onUpdateProgress(String packageName, double added,
					double downloaded, double total) {
				if (lastTime == 0) {
					lastTime = startTime;
				}
				final long nowTime = System.currentTimeMillis();
				final long castTime = nowTime - lastTime;
				if (castTime == 0 || added == 0) {
					lastTime = nowTime;
					return;
				}
				System.out.println("Downloaded Percent : "
						+ downloaded
						/ total
						+ " at speed : "
						+ SpeedDetectUtil.getSpeedString(
								castTime == 0 ? castTime + 1 : castTime,
								added == 0 ? (long)(added + 1) : (long)added));
				lastTime = nowTime;
			}
		});

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
			}

			@Override
			public void onFailure(String packageName, Exception e) {
				Assert.fail();
			}
		});
		downloadInfo.putInt(DownloadInfo.P_SECTION_COUNT, 1);
		dm.startDownload(downloadInfo);
		try {
			latcher.await();
		} catch (InterruptedException e1) {
			e1.printStackTrace();
			Assert.fail();
		}
	}

}
