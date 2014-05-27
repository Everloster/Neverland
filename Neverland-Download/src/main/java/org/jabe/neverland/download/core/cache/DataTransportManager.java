package org.jabe.neverland.download.core.cache;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.SynchronousQueue;

public class DataTransportManager {
	
	private DataTransportManager() {
	}
	
	private static DataTransportManager sInstance;
	
	public static DataTransportManager getInstance() {
		
		synchronized (DataTransportManager.class) {
			if (sInstance == null) {
				synchronized (sInstance) {
					sInstance = new DataTransportManager();
				}
			}
		}
		
		return sInstance;
	}
	
	@SuppressWarnings("unused")
	private Object mDataInLock = new Object();
	
	@SuppressWarnings("unused")
	private Object mDataOutLock = new Object();
	
	private Object mDataRegisterLock = new Object();
	
	private volatile boolean working = true;
	
	private Map<Integer, DataSpec> mDataSpecMap = Collections.synchronizedMap(new HashMap<Integer, DataSpec>());
	
	private Queue<DataBean> mDataBeanQueue = new SynchronousQueue<DataTransportManager.DataBean>();
	
	
	public void registerData(DataSpec dataSpec) {
		if (!mDataSpecMap.containsKey(dataSpec.hashCode())) {
			mDataSpecMap.put(dataSpec.hashCode(), dataSpec);
			mDataRegisterLock.notify();
		}
	}
	
	public void addData(DataBean dataBean) {
		mDataBeanQueue.add(dataBean);
	}
	
	@SuppressWarnings("unused")
	private class Worker implements Runnable {

		@Override
		public void run() {
			while(working) {
				if (!mDataSpecMap.isEmpty()) {
					if (mDataBeanQueue.size() != 0) {
						final DataBean bean = mDataBeanQueue.poll();
						if (bean != null) {
							// do some thing for bean
							
						}
					} else {
						try {
							Thread.sleep(3000);
						} catch (InterruptedException e) {
							e.printStackTrace();
						}
					}
				} else {
					try {
						Thread.sleep(5000);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
		
	}
	
	public static class DataBean {
		public String savePath;
		public int parentId;
		public byte[] data;
		public int position;
	}
	
	public static class DataSpec {
		public String savePath;
		public int id;
		public long length;
	}
	
	public static interface Callback {
		
		public void ouSuccess();
		
		public void onFailure();
	}
}
