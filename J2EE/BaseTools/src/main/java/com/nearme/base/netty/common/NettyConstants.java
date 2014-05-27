/**
 * NettyConstants.java
 * com.nearme.base.netty.common
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-4-26 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.common;
/**
 * ClassName:NettyConstants <br>
 * Function: 常量定义 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-4-26  下午09:45:57
 */
public class NettyConstants {
	/**
	 * 默认的启动端口
	 */
	public static final int DEFAULT_SERVER_PORT = 11216;
	
	/**
	 * 最小的服务端口
	 */
	public static final int MIN_SERVER_PORT = 1024;
	
	/**
	 * 最大的服务端口
	 */
	public static final int MAX_SERVER_PORT = 65535;
	
	/**
	 * 最小超时时间
	 */
	public static final long MIN_CLIENT_TIMEOUT = 1000;
	
	/**
	 * 请求发生未知错误
	 */
	public static final int REQUEST_UNKNOWN_ERROR = -1;
	
	/**
	 * 请求未成功
	 */
	public static final int REQUEST_NOT_SUCCESS = -2;
	
	/**
	 * 请求超时
	 */
	public static final int REQUEST_TIMEOUT = -3;
	
	/**
	 * 请求被取消
	 */
	public static final int REQUEST_CANCELLED = -4;
	
	/**
	 * 正常返回代码，小于此值表示存在服务器异常
	 */
	public static final int RESPONSE_NORMAL = 0;
	
	/**
	 * 未知的错误
	 */
	public static final int RESPONSE_UNKNOW_EXCEPTION = -1001;
	
	/**
	 * 空返回
	 */
	public static final int RESPONSE_RESPONSE_NULL = -1002;
	
	/**
	 * 不存在的业务类型
	 */
	public static final int RESPONSE_UNKNOW_TYPE = -1003;
	
	/**
	 * 不合法的数据协议
	 */
	public static final int RESPONSE_ILLEAGEL_PROTOCOL = -1004;
}

