/**
 * Copyright (C) 2014, all rights reserved.
 * Company	SHENZHEN YUNZHONGFEI TECHNOLOGY CORP., LTD. 
 * Author	lailong
 * Since	2014年1月21日
 */
package com.nearme.gamecenter.ddz.redis;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.exceptions.JedisConnectionException;

import com.nearme.gamecenter.ddz.oauth.common.BaseConfig;

/**
 * 
 * @Author LaiLong
 * @Since 2014年1月21日
 */
public class RedisServer {

	public final static String TIME_OUT_STRING = "TIME_OUT";

	private static final Logger logger = LoggerFactory
			.getLogger(RedisServer.class);

	private static RedisServer sInstance;

	private final ConcurrentHashMap<String, RedisCallback> mHandlerMap = new ConcurrentHashMap<String, RedisCallback>();

	private final static String REDIS_POP_LIST_NAME = "login_resp";

	private final static String REIDS_PUSH_LIST_NAME = "login_req";

	private final static long TIME_OUT = 10000;

	private Jedis mMainThreadJedis;

	private JedisPool mJedisPool;
	
	private RedisServer() {
		final String host = BaseConfig
				.getProjectConfig("redis-server-getuserinfo-host");
		final String port = BaseConfig
				.getProjectConfig("redis-server-getuserinfo-port");
		mJedisPool = new JedisPool(host, Integer.parseInt(port));
		mMainThreadJedis = mJedisPool.getResource();
	}

	public void startServer() {
		startMainWorker();
		startTimeoutWorker();
	}

	/**
	 * 
	 */
	private void startTimeoutWorker() {
		// do nothing
	}

	/**
	 * 
	 */
	private void startMainWorker() {
		final Thread mainThread = new Thread(new Runnable() {

			@SuppressWarnings("static-access")
			@Override
			public void run() {
				while (true) {
					String data = null;
					try {
						data = mMainThreadJedis.lpop(REDIS_POP_LIST_NAME);
					} catch (Exception e) {
						mMainThreadJedis.disconnect();
						logger.warn("can't connect to redis server : "
								+ e.getMessage());
						try {
							Thread.currentThread().sleep(10000);
						} catch (InterruptedException e2) {
						}
					}
					if (data != null) {
						logger.info("get redis server user info : " + data);
						try {
							final JSONObject jsonObject = new JSONObject(data);
							final String key = jsonObject.getString("msg_id");
							final RedisCallback callback = mHandlerMap.get(key);
							if (callback != null) {
								callback.fire(data);
								mHandlerMap.remove(key);
							}
						} catch (Exception e) {
							logger.warn("redis server get userinfo data error, data:"
									+ data + ", e:" + e.getMessage());
						}
						continue;
					}

					// check time out
					final long currentTime = System.currentTimeMillis();
					if (!mHandlerMap.isEmpty()) {
						Set<String> keys = mHandlerMap.keySet();
						for (String key : keys) {
							RedisCallback callback = mHandlerMap.get(key);
							if (currentTime - callback.getStartTimeMillis() > TIME_OUT) {
								callback.fire(TIME_OUT_STRING);
								mHandlerMap.remove(key);
								logger.warn("redis server get userinfo timeout, messge_id:"
										+ callback.getIdentify());
							}
						}
						continue;
					} else {
						try {
							Thread.currentThread().sleep(1000);
						} catch (InterruptedException e) {
						}
					}
				}
			}
		});
		mainThread.setDaemon(true);
		mainThread.setName("RedisMain");
		mainThread.setPriority(Thread.MAX_PRIORITY);
		mainThread.start();
	}

	public static RedisServer getInstance() {
		synchronized (RedisServer.class) {
			if (sInstance == null) {
				sInstance = new RedisServer();
			}
			return sInstance;
		}
	}

	public boolean addMessage(RedisCallback callback, String message) {
		Jedis jedis = mJedisPool.getResource();
		try {
			long result = jedis.lpush(REIDS_PUSH_LIST_NAME, message);
			if (result != 0) {
				if (mHandlerMap.contains(callback.getIdentify())) {
					return false;
				}
				mHandlerMap.put(callback.getIdentify(), callback);
//				testAdd(callback.getIdentify());
				return true;
			} else {
				logger.error("push message to reids failure : " + message);
				return false;
			}
		} catch (JedisConnectionException e) {
			// returnBrokenResource when the state of the object is
			// unrecoverable
			if (null != jedis) {
				mJedisPool.returnBrokenResource(jedis);
				jedis = null;
			}
		} finally {
			// / ... it's important to return the Jedis instance to the pool
			// once you've finished using it
			if (null != jedis)
				mJedisPool.returnResource(jedis);
		}
		return false;
	}

	private void testAdd(String messageId) {
		try {
			final JSONObject jsonObject = new JSONObject();
			jsonObject.put("msg_id", messageId);
			jsonObject.put("something", "hehe");
			mMainThreadJedis.lpush(REDIS_POP_LIST_NAME, jsonObject.toString());
		} catch (Exception e) {
		}
	}

}
