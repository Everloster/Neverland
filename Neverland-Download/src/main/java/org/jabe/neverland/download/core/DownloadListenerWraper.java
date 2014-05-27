package org.jabe.neverland.download.core;

public class DownloadListenerWraper implements DownloadRegister {
	
	private AbstractMessageDeliver mDeliver;
	
	public DownloadListenerWraper(AbstractMessageDeliver downloadMessageDeliver) {
		this.mDeliver = downloadMessageDeliver;
	}

	/* (non-Javadoc)
	 * @see org.jabe.neverland.download.core.DownloadInterface#registerListener(org.jabe.neverland.download.core.DownloadListener)
	 */
	@Override
	public void registerListener(DownloadListener downloadInterface) {
		mDeliver.registerListener(downloadInterface);
	}

	/* (non-Javadoc)
	 * @see org.jabe.neverland.download.core.DownloadInterface#removeListenter(org.jabe.neverland.download.core.DownloadListener)
	 */
	@Override
	public void removeListenter(DownloadListener downloadInterface) {
		mDeliver.removeListenter(downloadInterface);
	}

	
	
}
