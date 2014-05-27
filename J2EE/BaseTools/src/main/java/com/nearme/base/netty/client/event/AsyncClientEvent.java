/**
 * AsyncClientEvent.java
 * com.nearme.base.netty.client.event
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-9-28 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.netty.client.event;

import java.util.EventObject;

import com.nearme.base.netty.client.HttpInstanceInfo;
import com.nearme.base.netty.common.ProtobufWrapperProtocol.ResponseData;

/**
 * ClassName:AsyncClientEvent <br>
 * Function: 异步client事件 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-9-28  上午09:32:43
 */
public class AsyncClientEvent extends EventObject {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;

	private ResponseData response;

	public AsyncClientEvent(HttpInstanceInfo httpInstanceInfo, ResponseData response) {
		super(httpInstanceInfo);

		this.response = response;
	}

	public ResponseData getResponseData() {
		return this.response;
	}
}

