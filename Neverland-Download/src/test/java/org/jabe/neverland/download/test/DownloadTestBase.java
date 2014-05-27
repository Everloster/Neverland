/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年4月3日
 */
package org.jabe.neverland.download.test;

import org.jabe.neverland.download.core.DownloadConfig;
import org.jabe.neverland.download.core.DownloadInfo;
import org.jabe.neverland.download.core.DownloadManager;

/**
 * 
 * @Author LaiLong
 * @Since 2014年4月3日
 */
public class DownloadTestBase {

	/**
	 * 
	 */
	public DownloadTestBase() {

	}

	protected final static String ROOT_PATH = "C://Users//Administrator//Desktop//";

	protected DownloadManager dm;

	protected DownloadInfo downloadInfo;
	
	protected DownloadInfo downloadInfo2;
	protected DownloadInfo downloadInfo3;
	

	public void setUp() {
		Thread.currentThread().setName("thread-main");
		initDownInfo();
		initManager();
		cleanFile();
	}


	/**
	 * 
	 */
	private void cleanFile() {
		dm.getDownloadedFile(downloadInfo).delete();
		dm.getDownloadedFile(downloadInfo2).delete();
		dm.getDownloadedFile(downloadInfo3).delete();
	}


	/**
	 * 
	 */
	private void initDownInfo() {
		downloadInfo = new DownloadInfo();
		downloadInfo
				.setmDownloadUrl("http://storefs.nearme.com.cn/uploadFiles/Pfiles/201307/15/b569cf9a2af44dbbaa5919686856907d.apk?n=com.nearme.gamecenter_V3.2.2");
		downloadInfo.setmIconUrl("");
		downloadInfo.setmId("100");
		downloadInfo.setmName("GameCenter");
		downloadInfo.setmPackageName("com.nearme.gamecenter");
		downloadInfo.putInt(DownloadInfo.P_SECTION_COUNT, 2);
		
		downloadInfo2 = new DownloadInfo();
		downloadInfo2
				.setmDownloadUrl("http://storefs.nearme.com.cn/uploadFiles/Pfiles/201401/12/bb473d973faa408f8e350c17a73d7838.apk?n=com.rasoft.bubble_1.2097");
		downloadInfo2.setmIconUrl("");
		downloadInfo2.setmId("101");
		downloadInfo2.setmName("PopDragon");
		downloadInfo2.setmPackageName("com.rasoft.bubble");
		downloadInfo2.putInt(DownloadInfo.P_SECTION_COUNT, 2);
		
		
		downloadInfo3 = new DownloadInfo();
		downloadInfo3
				.setmDownloadUrl("http://storefs.nearme.com.cn/uploadFiles/Pfiles/201403/14/1754c4c29b3b42ff9a9be5a59b9160b4.apk?n=com.caimi.creditcard_4.1.5");
		downloadInfo3.setmIconUrl("");
		downloadInfo3.setmId("102");
		downloadInfo3.setmName("creditcard");
		downloadInfo3.setmPackageName("com.caimi.creditcard");
		downloadInfo3.putInt(DownloadInfo.P_SECTION_COUNT, 2);
	}


	/**
	 * 
	 */
	protected void initManager() {
		final DownloadConfig downloadConfig = new DownloadConfig.Builder()
				.addMaxTaskSizeInRunning(2)
				.addCacheRootPath(ROOT_PATH).build();
		dm = new DownloadManager(downloadConfig);
//		dm.registerListener(new DownloadStatusListener() {
//
//			@Override
//			public void onStatusCountChanged(int downloadingCount) {
//
//			}
//
//			@Override
//			public void onStatusChanged(String packageName, int status) {
//				if (status == DownloadStatus.DOWNLOAD_STATUS_FINISHED.ordinal()) {
//					System.out
//							.println(packageName + " has finish downloading.");
//				} else if (status == DownloadStatus.DOWNLOAD_STATUS_CANCEL
//						.ordinal()) {
//					System.out
//							.println(packageName + " has cancel downloading.");
//				}
//			}
//
//			@Override
//			public void onFailure(String packageName, Exception e) {
//				System.out.println(packageName + " download failure : "
//						+ e.getMessage());
//			}
//		});
//		dm.registerListener(new DownloadProgressListener() {
//
//			@Override
//			public void onUpdateProgress(String packageName, double added,
//					double downloaded, double total) {
//				System.out
//						.println("Downloaded Percent : " + downloaded / total);
//				System.out.println("changed by current thread "
//						+ Thread.currentThread().getName() + " " + downloaded
//						/ total);
//			}
//		});
	}

}
