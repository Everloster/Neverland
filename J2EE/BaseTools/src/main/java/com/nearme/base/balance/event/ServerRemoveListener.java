/**
 * ServerRemoveListener.java
 * com.nearme.base.balance.event
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-18 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.event;

import java.util.EventListener;

/**
 * ClassName:ServerRemoveListener <br>
 * Function: 服务被移除监听 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-8-18  上午10:50:44
 */
public interface ServerRemoveListener extends EventListener {
	//事件响应
    void onServerRemove(ServerRemoveEvent sre);
}

