/**
 * ClientAccessRequestHandler.java
 * com.nearme.market.access.main
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-21 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.gamecenter.ddz.oauth.server;

import com.nearme.gamecenter.ddz.oauth.common.RequestConfig;
import com.oppo.base.common.NumericUtil;

/**
 * ClassName:ClientAccessRequestHandler <br>
 * Function: 客户端访问的抽象处理类 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-9-21  下午04:13:53
 */
public class ClientAccessRequestHandler extends AbstractRequestHandler {
	
	private int requestType;	//业务时间
	private int timeout;		//超时设置
	
	private static final int DEFAULT_TIMEOUT = 10000;

	@Override
	public void init(RequestConfig config) {
		requestType = NumericUtil.parseInt(config.getExtension0(), 0);
		timeout = NumericUtil.parseInt(config.getExtension1(), DEFAULT_TIMEOUT);
	}
	
	/**
	 * 获取请求的类型id
	 * @param 
	 * @return
	 */
	public int getRequestType() {
		return requestType;
	}
	
	/**
	 * 接口的超时时间
	 * @param 
	 * @return
	 */
	public int getTimeout() {
		if(timeout <= 0) {
			timeout = DEFAULT_TIMEOUT;
		}
		
		return timeout;
	}
}

