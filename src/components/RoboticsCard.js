import React from 'react';
import clsx from 'clsx';
import styles from './RoboticsCard.module.css';

// Component for displaying robotics concepts with consistent styling
const RoboticsCard = ({ title, description, icon, children }) => {
  return (
    <div className={clsx('col col--4', styles.roboticsCard)}>
      <div className="card">
        <div className="card__header">
          <h3>{title}</h3>
          {icon && <div className={styles.icon}>{icon}</div>}
        </div>
        <div className="card__body">
          <p>{description}</p>
          {children}
        </div>
      </div>
    </div>
  );
};

export default RoboticsCard;