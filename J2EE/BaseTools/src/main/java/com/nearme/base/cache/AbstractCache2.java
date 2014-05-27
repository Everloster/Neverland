/**
 * AbstractCache2.java
 * com.nearme.base.cache
 *
 * Function： TODO
 *
 *   ver     date      		author
 * ──────────────────────────────────
 *   1.0	 2011-10-19 		80036381
 *
 * Copyright (c) 2011 NearMe, All Rights Reserved.
*/

package com.nearme.base.cache;

import com.nearme.base.cache.selector.CacheSelector;
import com.oppo.base.cache.util.CacheStatistic;

/**
 * ClassName:AbstractCache2
 * Function: ICache2的抽象实现类，实现了其基本的操作流程
 * 缓存更新时,应先更新下一层的缓存,然后更新当前缓存
 * 缓存查询时,应先从本层获取,未获取到才考虑从下层获取
 *
 * @author   80036381
 * @version
 * @since    Ver 1.1
 * @Date	 2011-10-19  上午09:52:38
 */
public abstract class AbstractCache2 implements ICache2 {
	private ICache2 nextCache;		//当前层的下一层
	private boolean isTierClosed;	//当前层是否关闭

	/**
	 * 获取数据
	 * @see com.nearme.base.cache.ICache2#get(com.nearme.base.cache.CacheKey, com.nearme.base.cache.selector.CacheSelector, long)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public <T> T get(CacheSelector cacheSelector) throws Exception {
		//先从本层取
		T myCache = null;
		if(!isTierClosed) {
			try{
				//防止从本层缓存读取出错时不能继续访问下面的层,加try catch
				myCache = (T)getCache(cacheSelector);
			}catch(Exception e){
				//如果不存在下层则报错
				if(null == this.nextCache) {
					throw e;
				}
			}

			if(null != myCache) {
				//统计命中并返回
				CacheStatistic.getInstance().addHit(this.getClass().getName());

				return myCache;
			} else if(null == this.nextCache) {
				//如果该层是最后一层且未命中，则当存在预设值时传回预设值
				Object emptyValue = cacheSelector.getEmptyValue();
				if(null != emptyValue) {
					myCache = (T)emptyValue;
				}
			}
		}

		//本层未获取到尝试从下层获取
		if(null == myCache && null != this.nextCache){
			myCache = (T)this.nextCache.get(cacheSelector);
		}

		//如果从下一层获取到了缓存并且本层缓存未关闭,则在本层也进行更新
		if(null != myCache && !isTierClosed) {
			try {
				//防止设置本层缓存出错,加try catch
				setCache(cacheSelector, myCache);
			} catch(Exception ex) {
				//如果不存在下层则报错
				if(null == this.nextCache) {
					throw ex;
				}
			}
		}

		return myCache;
	}

	/**
	 * 更新数据
	 * @see com.nearme.base.cache.ICache2#update(com.nearme.base.cache.CacheKey, com.nearme.base.cache.selector.CacheSelector, long)
	 */
	@Override
	public int update(CacheSelector cacheSelector) throws Exception {
		if(null != this.nextCache){

			int affRow = this.nextCache.update(cacheSelector);
			//清除本层缓存
			deleteCache(cacheSelector);

			return affRow;
		}

		if(this.isTierClosed) { //如果本层关闭则直接返回失败
			return 0;
		} else {
			return updateCache(cacheSelector);
		}
	}

	/**
	 * 添加数据
	 * @see com.nearme.base.cache.ICache2#insert(com.nearme.base.cache.CacheKey, com.nearme.base.cache.selector.CacheSelector, long)
	 */
	@Override
	public int insert(CacheSelector cacheSelector) throws Exception {
		int insertCount = 0;
		if(null != this.nextCache){
			insertCount = this.nextCache.insert(cacheSelector);

			if(insertCount > 0) { //添加成功后删除对应缓存--一个key对应一组对象的情况
				this.deleteCache(cacheSelector);
			}
		} else if(this.isTierClosed) {//最后一层关闭则无法进行保存
			throw new Exception(this.getClass().getName() + " is closed, insert fail.");
		} else {
			insertCount = insertCache(cacheSelector);
		}

		return insertCount;
	}

	/**
	 * 删除数据
	 * @see com.nearme.base.cache.ICache2#delete(com.nearme.base.cache.CacheKey, com.nearme.base.cache.selector.CacheSelector, long)
	 */
	@Override
	public int delete(CacheSelector cacheSelector) throws Exception {
		int deleteCount = 0;

		if(null != this.nextCache) {
			//先删除底层的缓存
			deleteCount = this.nextCache.delete(cacheSelector);
		}

		if(!this.isTierClosed) {
			int thisAffRow = deleteCache(cacheSelector);
			//如果这一层是最后一层，则以该层的删除数为主
			if(null == this.nextCache) {
				deleteCount = thisAffRow;
			}
		}

		return deleteCount;
	}

	/**
	 * 资源释放
	 * @see com.nearme.base.cache.ICache2#release()
	 */
	@Override
	public void release() {
		if(null != this.nextCache) {
			this.nextCache.release();
		}

		releaseResource();
	}

	/**
	 * 获取本层缓存信息
	 */
	protected abstract Object getCache(CacheSelector cacheSelector) throws Exception;

	/**
	 * 设置本层缓存操作
	 * @param cacheSelector
	 * @param timeout
	 * @return
	 */
	protected abstract boolean setCache(CacheSelector cacheSelector, Object cacheValue) throws Exception;


	/**
	 * 添加数据操作
	 * @param cacheSelector
	 * @param timeout
	 * @return
	 */
	protected abstract int insertCache(CacheSelector cacheSelector) throws Exception;

	/**
	 * 更新数据操作
	 * @param cacheSelector
	 * @param timeout
	 * @return
	 */
	protected abstract int updateCache(CacheSelector cacheSelector) throws Exception;

	/**
	 * 删除本层缓存
	 * @param cacheSelector
	 * @param timeout
	 * @return
	 */
	protected abstract int deleteCache(CacheSelector cacheSelector) throws Exception;

	/**
	 * 资源释放
	 * @param
	 * @return
	 */
	public abstract void releaseResource();

	/**
	 * 获取下一层缓存
	 * @return
	 */
	public ICache2 getNextCache() {
		return nextCache;
	}

	/**
	 * 设置下一层缓存
	 * @param nextCache
	 */
	public void setNextCache(ICache2 nextCache) {
		if(this.expireSupported()) {
			this.nextCache = nextCache;
		} else {
			throw new UnsupportedOperationException(this.getClass().getName()
					+ " not support this method setNextCache because it is not support date expire.");
		}
	}

	/**
	 * 获取本层是否启用
	 * @return
	 */
	public boolean isTierClosed() {
		return this.isTierClosed;
	}

	/**
	 * 设置本层是否启用
	 * @param isTierClosed
	 */
	public void setTierClosed(boolean isTierClosed) {
		this.isTierClosed = isTierClosed;
	}
}

