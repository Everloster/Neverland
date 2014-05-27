/**
 * ServerRemoveEvent.java
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

import java.util.EventObject;

import com.nearme.base.balance.model.ExecuteInfo;
import com.nearme.base.balance.model.ServerInfo;

/**
 * ClassName:ServerRemoveEvent <br>
 * Function: 服务被移除事件 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-8-18  上午10:48:54
 */
public class ServerRemoveEvent extends EventObject {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private transient ExecuteInfo executeInfo;
	
	public ServerRemoveEvent(ServerInfo source, ExecuteInfo executeInfo) {
		super(source);
		
		this.executeInfo = executeInfo;
	}
	
	/**
	 * 获取执行统计信息
	 * @param 
	 * @return
	 */
	public ExecuteInfo getExecuteInfo() {
		return this.executeInfo;
	}
}

