package org.jabe.neverland.model;

import android.view.View;

public abstract class ViewHolder implements IViewHolder {

	public abstract void saveView(View view, int position, boolean save);
	
	protected void save(View view, Object o, Boolean save) {
		if (save) {
			if (view != null) {
				view.setTag(o);
			}
		}
	}
}
