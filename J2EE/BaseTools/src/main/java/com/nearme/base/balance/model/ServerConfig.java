/**
 * ServerConfig.java
 * com.nearme.base.balance.model
 *
 * Function： TODO 
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-15 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance.model;

import java.io.Serializable;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ClassName:ServerConfig <br>
 * Function: 服务配置 <br>
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2012-8-15  下午06:11:07
 */
public class ServerConfig implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;
	
	private Map<String, ServerGroupInfo> serverGroups;

	/**
	 * 获取serverGroups
	 * @return  the serverGroups
	 * @since   Ver 1.0
	 */
	public Map<String, ServerGroupInfo> getServerGroups() {
		return serverGroups;
	}

	/**
	 * 设置serverGroups
	 * @param   serverGroups    
	 * @since   Ver 1.0
	 */
	public void setServerGroups(Map<String, ServerGroupInfo> serverGroups) {
		if(null != serverGroups && !(serverGroups instanceof ConcurrentHashMap)) {
			this.serverGroups = new ConcurrentHashMap<String, ServerGroupInfo>(serverGroups);
		} else {
			this.serverGroups = serverGroups;
		}
	}
	
	/**
	 * 添加组信息，以组id为key
	 * @param 
	 * @return
	 */
	public void addServerGroup(ServerGroupInfo groupInfo) {
		if(null == this.serverGroups) {
			this.serverGroups = new ConcurrentHashMap<String, ServerGroupInfo>();
		}
		
		this.serverGroups.put(groupInfo.getId(), groupInfo);
	}
	
	public ServerGroupInfo getServerGroup(String groupId) {
		if(null == this.serverGroups) {
			return null;
		} else {
			return this.serverGroups.get(groupId);
		}
	}
}

