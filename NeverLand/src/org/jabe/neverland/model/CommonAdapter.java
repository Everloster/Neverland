package org.jabe.neverland.model;

import java.util.List;

import org.jabe.neverland.util.Util;

import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;

public class CommonAdapter<T> extends BaseAdapter{
	
	public CommonAdapter(List<ListElement<T>> mList) {
		super();
		this.mList = mList;
	}

	private List<ListElement<T>> mList;

	@Override
	public int getCount() {
		// TODO Auto-generated method stub
		return mList.size();
	}

	@Override
	public ListElement<T> getItem(int position) {
		return mList.get(position);
	}

	@Override
	public long getItemId(int position) {
		return position;
	}

	@Override
	public View getView(int position, View convertView, ViewGroup parent) {
		return mList.get(position).getView(position, convertView, parent);
	}

	public void remove(ListElement<T> dragItem, int position) {
		dragItem.remove(position);
		mList.remove(position);
	}

	public void insert(ListElement<T> dragItem, int position, T data) {
		dragItem.insert(data, position);
		mList.add(position, dragItem);
	}
	
	/**
	 * clone第一个,加入开头
	 * @param t
	 */
	public void addFirst(T t){
		ListElement<T> dragItem = (ListElement<T>) this.getItem(0);
		if (dragItem.canReuse()) {
			this.insert(dragItem.clone(), 0, t);
		} else {
			Util.dout("this item can not reuse so can add");
		}
		
	}
	
	/**
	 * 开头插入Item
	 * @param t
	 * @param dragItem
	 */
	public void addFirst(T t, ListElement<T> dragItem){
		this.insert(dragItem.clone(), 0, t);
	}
	
	/**
	 * clone第一个,加入末尾
	 * @param t
	 */
	public void addEnd(T t){
		ListElement<T> dragItem = (ListElement<T>) this.getItem(0);
		if (dragItem.canReuse()) {
			dragItem.add(t);
			mList.add(dragItem.clone());
		} else {
			Util.dout("this item can not reuse so can not add");
		}
	}

	public void addEnd(T t, ListElement<T> dragItem){
		dragItem.add(t);
		mList.add(dragItem.clone());
	}
}
