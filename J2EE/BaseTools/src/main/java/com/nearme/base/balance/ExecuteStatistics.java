/**
 * ExecuteStatistics.java
 * com.nearme.base.balance
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2012-8-15 		80036381
 *
 * Copyright (c) 2012 NearMe, All Rights Reserved.
*/

package com.nearme.base.balance;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;

import com.nearme.base.balance.event.ServerRemoveEvent;
import com.nearme.base.balance.event.ServerRemoveListener;
import com.nearme.base.balance.model.ExecuteInfo;
import com.nearme.base.balance.model.ServerGroupInfo;
import com.nearme.base.balance.model.ServerInfo;

/**
 * ClassName:ExecuteStatistics <br>
 * Function: 执行统计 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-8-15  下午09:13:51
 */
public class ExecuteStatistics {

	private static final ExecuteStatistics instance = new ExecuteStatistics();

	private ConcurrentHashMap<String, ConcurrentHashMap<String, ExecuteInfo>> executeMap =
		new ConcurrentHashMap<String, ConcurrentHashMap<String, ExecuteInfo>>();
	private ReentrantLock lock = new ReentrantLock();//同步锁

	//服务移除监听器
	private List<ServerRemoveListener> serverRemoveListeners = new ArrayList<ServerRemoveListener>();

	private ExecuteStatistics() {
	}

	public static ExecuteStatistics getInstance() {
		return instance;
	}

	/**
	 * 添加服务移除监听器
	 * @param
	 * @return
	 */
	public void addServerRemoveListener(ServerRemoveListener serverRemoveListener) {
		serverRemoveListeners.add(serverRemoveListener);
	}

	/**
	 * 移除服务移除监听器
	 * @param
	 * @return
	 */
	public void removeServerRemoveListener(ServerRemoveListener serverRemoveListener) {
		serverRemoveListeners.remove(serverRemoveListener);
	}

	/**
	 * 添加执行结果
	 * @param
	 * @return
	 */
	public void addExecuteResult(String groupId, ServerInfo server, boolean success) {
		//找到对应的map
		ConcurrentHashMap<String, ExecuteInfo> singleGroupMap = executeMap.get(groupId);
		if(null == singleGroupMap) {
			singleGroupMap = new ConcurrentHashMap<String, ExecuteInfo>();
			ConcurrentHashMap<String, ExecuteInfo> oldMap = executeMap.putIfAbsent(groupId, singleGroupMap);
			if(null != oldMap) {
				singleGroupMap = oldMap;
			}
		}

		//找到对应的执行信息进行更新
		String serverId = server.getServerId();
		ExecuteInfo eInfo = null;
		try {
			lock.lock();
			eInfo = singleGroupMap.get(serverId);
			boolean create = false;
			if(null == eInfo) {
				eInfo = new ExecuteInfo(server);
				create = true;
			}

			if(success) {
				eInfo.addSuccess();
			} else {
				eInfo.addFail();
			}

			if(create) {
				singleGroupMap.put(serverId, eInfo);
			}
		} finally {
			lock.unlock();
		}

		//更新完成后计算是否需要调整权重,是否需要移除失败率高的服务器
		if(null != eInfo) {
			ServerGroupInfo group = server.getServerGroup();
			if(eInfo.getTotal() >= group.getAbandonMinSize()) {
				//达到最小丢弃数量后判断失败率
				double failRate = eInfo.getFail() * 1.0 / eInfo.getTotal();
				if(failRate >= group.getAbandonRate()) {
					//从组中移除
					if(group.removeServer(server)) {
						//重置统计信息
						eInfo.reset();
						onServerRemove(server, eInfo);
					}
				}
			}
		}
	}

	//通知服务被移除的事件
	private void onServerRemove(ServerInfo server, ExecuteInfo executeInfo) {
		List<ServerRemoveListener> listenerList = new ArrayList<ServerRemoveListener>(serverRemoveListeners);
		int size = listenerList.size();
		if(size > 0) {
			ServerRemoveEvent sre = new ServerRemoveEvent(server, executeInfo);
			for(int i = 0; i < size; i++) {
				listenerList.get(i).onServerRemove(sre);
			}
		}
	}
}

