package com.oppo.base.cache;

import com.oppo.base.cache.util.CacheStatistic;

/**
 * ClassName:AbstractCache
 * Function: 缓存抽象类,每一层缓存更新时,应先更新下一层的缓存,若下一层缓存更新成功则删除缓存
 *
 * @author   80036381
 * @version  
 * @since    Ver 1.1
 * @Date	 2011-6-8  上午10:58:29
 */
public abstract class AbstractCache implements ICache {
	private ICache nextCache;
	private boolean isTierClosed;
	private boolean isInsertSave;

	public AbstractCache() {
		this.isTierClosed = false;
		this.isInsertSave = true;
	}

	public int delete(String cacheId, StorageObject cacheObject, long timeout) throws Exception{
		int affRow = 0;
		
		if(null != this.nextCache) { 
			//先删除底层的缓存
			affRow = this.nextCache.delete(cacheId, cacheObject, timeout);
		}
		
		if(!this.isTierClosed) {
			int thisAffRow = deleteCache(cacheId, cacheObject, timeout);
			//如果这一层是最后一层，则以该层的删除数为主
			if(null == this.nextCache) {
				affRow = thisAffRow;
			}
		}
		
		return affRow;
	}

	public Object get(String cacheId, StorageObject cacheObject, long timeout)  throws Exception{
		//先从本层取
		Object myCache = null;
		if(!isTierClosed) {
			try{
				//防止从本层缓存读取出错时不能继续访问下面的层,加try catch
				myCache = getCache(cacheId, cacheObject);	
			}catch(Exception e){
				//如果不存在下层则报错
				if(null == this.nextCache) {
					throw e;
				}
			}
			if(null != myCache) {//命中统计
				CacheStatistic.getInstance().addHit(this.getClass().getName());
			}
		}
		
		//如果缓存未关闭,或者已经没有一下层,则
		if(null == myCache && null != this.nextCache){
			myCache = this.nextCache.get(cacheId, cacheObject, timeout);
			if(null != myCache && !isTierClosed) { 
				//如果从下一层获取到了缓存并且缓存未关闭,则在本级也进行更新
				setCache(cacheId, myCache, timeout);
			}
		}
		
		return myCache;
	}
	
	public int insert(String cacheId, StorageObject cacheObject, long timeout) throws Exception{
		int insertCount = 0;
		if(null != this.nextCache){
			insertCount = this.nextCache.insert(cacheId, cacheObject, timeout);
			
			if(insertCount > 0) { //添加成功后删除对应缓存--一个key对应一组对象的情况
				this.deleteCache(cacheId, cacheObject, timeout);
			}
		} else if(this.isTierClosed) {//最后一层关闭则无法进行保存
			throw new Exception(this.getClass().getName() + " is closed, insert fail.");
		} else {
			insertCount = insertCache(cacheId, cacheObject, timeout);
		}
		
		/* 删除by 阳波2011-08-22
		//插入成功后也可能返回null,因此暂时去掉第一个限制
		if(!this.isTierClosed 
				&& !StringUtil.isNullOrEmpty(cacheId)
				&& this.isInsertSave() ){ //如果插入成功并且需要保存cache
			try{
				//设置失败不影响结果，因此加try catch
				this.setCache(cacheId, cacheObject.getStorageObject(), timeout);
			} catch(Exception e) {
			}
		}
		*/
		
		return insertCount;
	}
	
	public int update(String cacheId, StorageObject cacheObject, long timeout) throws Exception{
		if(null != this.nextCache){
			int affRow = this.nextCache.update(cacheId, cacheObject, timeout);
			//if(affRow > 0 && !this.isTierClosed) { //更新成功则清除本层缓存
			deleteCache(cacheId, cacheObject, 0);
			//}
			
			return affRow;
		}
		
		if(this.isTierClosed) { //如果本层关闭则直接返回失败
			return 0;
		} else {
			return updateCache(cacheId, cacheObject, timeout);
		}
	}
	
	@Override
	public void release() {
		if(null != this.nextCache) {
			this.nextCache.release();
		}
		
		releaseResource();
	}
	
	/**
	 * 获取下一层缓存
	 * @return
	 */
	public ICache getNextCache() {
		return nextCache;
	}

	/**
	 * 设置下一层缓存
	 * @param nextCache
	 */
	public void setNextCache(ICache nextCache) {
		this.nextCache = nextCache;
	}
	
	/**
	 * 获取本层是否启用
	 * @return
	 */
	public boolean isTierClosed() {
		return isTierClosed;
	}

	/**
	 * 设置本层是否启用
	 * @param isTierClosed
	 */
	public void setTierClosed(boolean isTierClosed) {
		this.isTierClosed = isTierClosed;
	}
	
	/**
	 * @deprecated 不再使用
	 * 是否将插入的值写入缓存
	 * Function Description here
	 * @param 
	 * @return
	 */
	public boolean isInsertSave() {
		return isInsertSave || null == this.nextCache;
	}

	/**
	 * @deprecated 不再使用
	 * 设置在insert数据时是否需要将数据保存到缓存中
	 * Function Description here
	 * @param 
	 * @return
	 */
	public void setInsertSave(boolean isInsertSave) {
		this.isInsertSave = isInsertSave;
	}
	

	/**
	 * 获取本层缓存信息
	 */
	public abstract Object getCache(String cacheId, StorageObject cacheObject) throws Exception;
	
	/**
	 * 设置本层缓存
	 * @param cacheId
	 * @param obj
	 * @return
	 */
	public abstract boolean setCache(String cacheId, Object obj, long timeout) throws Exception;

	
	/**
	 * 添加数据(缓存层返回null)
	 * @param sqlObject
	 * @return
	 */
	public abstract int insertCache(String cacheId, StorageObject cacheObject, long timeout) throws Exception;
	
	/**
	 * 更新数据(缓存层返回0)
	 * @param sql
	 * @return
	 */
	public abstract int updateCache(String cacheId, StorageObject cacheObject, long timeout) throws Exception;
	
	/**
	 * 删除本层缓存
	 * @param cacheId
	 * @param sqlObject
	 * @param timeout
	 * @return
	 */
	public abstract int deleteCache(String cacheId, StorageObject cacheObject, long timeout) throws Exception;

	/**
	 * 资源释放
	 * @param 
	 * @return
	 */
	public abstract void releaseResource();
}
