/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年3月12日
 */
package org.jabe.neverland.download.core;

import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadFactory;

/**
 * 
 * @Author LaiLong
 * @Since 2014年3月12日
 */
public class DefaultMessageDeliver extends AbstractMessageDeliver {

	private LinkedBlockingQueue<Message> queue = new LinkedBlockingQueue<Message>();

	private volatile boolean isRunning = false;

	/**
	 * 
	 */
	public DefaultMessageDeliver() {
		start();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * org.jabe.neverland.download.core.AbstractMessageDeliver#push(org.jabe
	 * .neverland.download.core.AbstractMessageDeliver.Message)
	 */
	@Override
	protected void push(Message message) {
		// whether the engine has fire the message.
		if (!fireMessageToEngine(message)) {
			queue.add(message);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.jabe.neverland.download.core.AbstractMessageDeliver#start()
	 */
	@Override
	public void start() {
		isRunning = true;
		Executors.newFixedThreadPool(1, new ThreadFactory() {

			@Override
			public Thread newThread(Runnable r) {
				final Thread th = new Thread(r);
				th.setName("thread-" + DefaultMessageDeliver.class.getSimpleName());
				th.setPriority(Thread.MAX_PRIORITY);
				return th;
			}
		}).execute(new Runnable() {

			@Override
			public void run() {
				while (isRunning) {
					try {
						final Message m = queue.take();
						fireMessageToClient(m);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}

		});
	}

	@Override
	public void shutdown() {
		isRunning = false;
		queue.clear();
	}


}
