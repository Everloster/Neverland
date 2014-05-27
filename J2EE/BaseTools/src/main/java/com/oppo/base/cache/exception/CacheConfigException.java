package com.oppo.base.cache.exception;

/**
 * ClassName:CacheStatistic
 * Function: 统计配置异常
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午10:58:29
 */
public class CacheConfigException extends Exception {
	private static final long serialVersionUID = -654361190647429494L;

	public CacheConfigException() {
		super();
	}

	public CacheConfigException(String message) {
		super(message);
	}

	public CacheConfigException(String message, Throwable cause) {
		super(message, cause);
	}

	public CacheConfigException(Throwable cause) {
		super(cause);
	}
}
