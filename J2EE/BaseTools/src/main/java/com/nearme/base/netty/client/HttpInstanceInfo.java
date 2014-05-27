/**
 * HttpInstanceInfo.java
 * com.nearme.base.netty.client
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-27 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client;

import java.util.List;

import com.nearme.base.netty.client.event.AsyncClientEvent;
import com.nearme.base.netty.client.event.AsyncClientListener;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.RequestData;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;

/**
 * ClassName:HttpInstanceInfo <br>
 * Function: http数据实例 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-9-27  下午05:47:52
 */
public class HttpInstanceInfo extends NettyHttpContext<AsyncClientListener> {
	private static final long serialVersionUID = 1L;

	private RequestData requestData;

	public HttpInstanceInfo() {
		super();
	}

	/**
	 * 服务端数据返回通知
	 * @param
	 * @return
	 */
	public void onResponse(ResponseData response) {
		this.setRequestComplete(true);

		List<AsyncClientListener> asyncClientListeners = getListeners();
		if(null == asyncClientListeners) {
			return;
		}

		int size = asyncClientListeners.size();
		if(size == 0) {
			return;
		}

		AsyncClientEvent sre = new AsyncClientEvent(this, response);
		for(int i = 0; i < size; i++) {
			asyncClientListeners.get(i).onServerResponse(sre);
		}
	}

	/**
	 * 获取requestData
	 * @return  the requestData
	 * @since   Ver 1.0
	 */
	public RequestData getRequestData() {
		return requestData;
	}
	/**
	 * 设置requestData
	 * @param   requestData
	 * @since   Ver 1.0
	 */
	public void setRequestData(RequestData requestData) {
		this.requestData = requestData;
	}
}

