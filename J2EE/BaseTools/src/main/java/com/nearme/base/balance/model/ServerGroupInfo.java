/**
 * ServerGroupInfo.java
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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantLock;

/**
 * ClassName:ServerGroupInfo <br>
 * Function: 服务组信息 <br>
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2012-8-15  下午04:59:14
 */
public class ServerGroupInfo implements Serializable {
	/**
	 * @since Ver 1.1
	 */
	private static final long serialVersionUID = 1L;

	private String id;
	private String description;
	private int abandonMinSize;	//抛弃最小值,当小于此值时不抛弃
	private double abandonRate;	//抛弃比例，当失败率达到此比例时，服务将被抛弃
	private boolean autoRestart;//是否自动重启

	private int totalWeight;
	private List<ServerInfo> serverList;
	private ReentrantLock lock;	//对添加或移除server加锁，防止并发操作产生异常

	public static final int DEFAULT_ABANDON_MIN_SIZE = 1000;
	public static final double DEFAULT_ABANDON_RATE = 0.9;

	public ServerGroupInfo() {
		abandonMinSize = DEFAULT_ABANDON_MIN_SIZE;
		abandonRate = DEFAULT_ABANDON_RATE;
		autoRestart = false;
		lock = new ReentrantLock();
	}

	/**
	 * 获取id
	 * @return  the id
	 * @since   Ver 1.0
	 */
	public String getId() {
		return id;
	}
	/**
	 * 设置id
	 * @param   id
	 * @since   Ver 1.0
	 */
	public void setId(String id) {
		this.id = id;
	}
	/**
	 * 获取description
	 * @return  the description
	 * @since   Ver 1.0
	 */
	public String getDescription() {
		return description;
	}
	/**
	 * 设置description
	 * @param   description
	 * @since   Ver 1.0
	 */
	public void setDescription(String description) {
		this.description = description;
	}

	/**
	 * 获取抛弃比例，当失败率达到此比例时，服务将被抛弃
	 * @return  the abandonRate
	 * @since   Ver 1.0
	 */
	public double getAbandonRate() {
		return abandonRate;
	}

	/**
	 * 设置抛弃比例，当失败率达到此比例时，服务将被抛弃
	 * @param   abandonRate
	 * @since   Ver 1.0
	 */
	public void setAbandonRate(double abandonRate) {
		if(abandonRate <= 0 || abandonRate > 1) {
			this.abandonRate = DEFAULT_ABANDON_RATE;
		} else {
			this.abandonRate = abandonRate;
		}
	}

	/**
	 * 获取抛弃最小值,当执行次数小于此值时不抛弃
	 * @return  the abandonMinSize
	 * @since   Ver 1.0
	 */
	public int getAbandonMinSize() {
		return abandonMinSize;
	}

	/**
	 * 设置抛弃最小值,当执行次数小于此值时不抛弃
	 * @param   abandonMinSize
	 * @since   Ver 1.0
	 */
	public void setAbandonMinSize(int abandonMinSize) {
		if(abandonMinSize < 1) {
			this.abandonMinSize = DEFAULT_ABANDON_MIN_SIZE;
		} else {
			this.abandonMinSize = abandonMinSize;
		}
	}

	/**
	 * 是否自动重启，如果为true则当group中的serverList都失效后将重新加载，
	 * 否则失效后则关闭相关服务
	 * @return  the autoRestart
	 * @since   Ver 1.0
	 */
	public boolean isAutoRestart() {
		return autoRestart;
	}

	/**
	 * 设置是否自动重启，如果为true则当group中的serverList都失效后将重新加载，
	 * 否则失效后则关闭相关服务
	 * @param   autoRestart
	 * @since   Ver 1.0
	 */
	public void setAutoRestart(boolean autoRestart) {
		this.autoRestart = autoRestart;
	}

	/**
	 * 获取serverList
	 * @return  the serverList
	 * @since   Ver 1.0
	 */
	public List<ServerInfo> getServerList() {
		return serverList;
	}
	/**
	 * 设置serverList
	 * @param   serverList
	 * @since   Ver 1.0
	 */
	public void setServerList(List<ServerInfo> serverList) {
		lock.lock();
		try {
			if(null != serverList) {
				for(int i = 0, size = serverList.size(); i < size; i++) {
					serverList.get(i).setServerGroup(this);
				}
			}

			this.serverList = serverList;
			refresh();
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 获取server个数
	 * @param
	 * @return
	 */
	public int getServerCount() {
		return null == serverList ? 0 : serverList.size();
	}

	/**
	 * 添加一个服务
	 * @param
	 * @return
	 */
	public void addServer(ServerInfo server) {
		if(null == server) {
			return;
		}

		lock.lock();
		try {
			if(null == serverList) {
				serverList = new ArrayList<ServerInfo>();
			}

			server.setServerGroup(this);
			serverList.add(server);
			refresh();
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 移除一个服务
	 * @param
	 * @return 如果server存在且被移除则返回true,否则返回false
	 */
	public boolean removeServer(ServerInfo server) {
		if(null == server || null == serverList) {
			return false;
		}

		lock.lock();
		try {
			boolean removed = serverList.remove(server);
			if (removed) {
				refresh();

				// 移除后将server状态设置为未激活的
				server.setActive(false);
			}

			return removed;
		} finally {
			lock.unlock();
		}
	}

	/**
	 * 获取totalWeight
	 * @return  the totalWeight
	 * @since   Ver 1.0
	 */
	public int getTotalWeight() {
		return totalWeight;
	}

	/**
	 * 刷新参数
	 * Function Description here
	 * @param
	 * @return
	 */
	public void refresh() {
		if(null == serverList) {
			this.totalWeight = 0;
		} else {
			for(int i = 0, size = serverList.size(); i < size; i++) {
				this.totalWeight += serverList.get(i).getWeight();
			}
		}
	}
}

