package org.jabe.neverland.pattern;

import java.util.ArrayList;
import java.util.List;

public class ObserverPattern {

	public interface DownloadListener {

		public void dispatchChange(String changeInfo);

	}

	public static abstract class DownloadManager {

		public abstract void registListener(DownloadListener downloadListener);

		public abstract void unregistListener(DownloadListener downloadListener);
		
	}
	
	public static class PageOne implements DownloadListener {

		@Override
		public void dispatchChange(String changeInfo) {
			System.out.println("PageOne receive change info : " + changeInfo);
		}
	}
	
	public static class PageTwo implements DownloadListener {

		@Override
		public void dispatchChange(String changeInfo) {
			System.out.println("PageTwo receive change info : " + changeInfo);
		}
	}
	
	public static class MutliThreadDownloadManager extends DownloadManager {

		private List<DownloadListener> mListeners = new ArrayList<ObserverPattern.DownloadListener>();

		@Override
		public void registListener(DownloadListener downloadListener) {
			mListeners.add(downloadListener);
		}

		@Override
		public void unregistListener(DownloadListener downloadListener) {
			mListeners.remove(downloadListener);
		}
		
		public void dispatchChange(String changeInfo) {
			for (DownloadListener downloadListener : mListeners) {
				downloadListener.dispatchChange(changeInfo);
			}
		}
		
	}
	
	public static void main(String[] args) {
		
		MutliThreadDownloadManager mDownloadManager = new MutliThreadDownloadManager();
		DownloadListener pageOne = new PageOne();
		DownloadListener pageTwo = new PageTwo();
		mDownloadManager.registListener(pageOne);
		mDownloadManager.registListener(pageTwo);
		mDownloadManager.dispatchChange("for test");
		
	}
}
