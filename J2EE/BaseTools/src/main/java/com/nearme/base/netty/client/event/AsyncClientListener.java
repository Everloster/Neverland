/**
 * AsyncClient.java
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

import java.util.EventListener;

/**
 * ClassName:AsyncClient <br>
 * Function: 异步client事件监听 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-9-28  上午09:38:44
 */
public interface AsyncClientListener extends EventListener {
	/**
	 * 返回数据事件
	 * @param
	 * @return
	 */
	void onServerResponse(AsyncClientEvent sre);
}

